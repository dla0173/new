"""포트폴리오 리밸런싱 시뮬레이터.

월별 수익률(또는 가격) 데이터로 여러 리밸런싱 전략의 성과를 비교한다.

전략
- none        : 최초 매수 후 보유(Buy & Hold), 리밸런싱 없음
- periodic    : N개월마다 목표 비중으로 리밸런싱 (월/분기/반기/연)
- threshold   : 어떤 자산이든 목표 비중에서 ±band(%p) 이상 벗어나면 리밸런싱

거래비용은 매매금액(매수+매도 합계)에 비례하는 비율(bp)로 차감한다.
세금, 슬리피지, 최소 거래단위는 반영하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


@dataclass
class Strategy:
    name: str
    kind: str  # "none" | "periodic" | "threshold"
    period_months: int = 0
    band: float = 0.0  # 절대 비중 차이, 예: 0.05 = 5%p

    def should_rebalance(self, month_index: int, weights: np.ndarray, target: np.ndarray) -> bool:
        if self.kind == "none":
            return False
        if self.kind == "periodic":
            return (month_index + 1) % self.period_months == 0
        if self.kind == "threshold":
            return bool(np.max(np.abs(weights - target)) > self.band + 1e-12)
        raise ValueError(f"알 수 없는 전략 종류: {self.kind}")


@dataclass
class SimulationResult:
    strategy: str
    values: pd.Series
    weights: pd.DataFrame
    rebalance_dates: list = field(default_factory=list)
    turnover: float = 0.0  # 누적 매매금액 / 해당 시점 평가금액의 합 (편도 기준)
    total_cost: float = 0.0  # 누적 거래비용 (금액)


def _rebalance(holdings: np.ndarray, target: np.ndarray, cost_rate: float) -> tuple[np.ndarray, float, float]:
    """목표 비중으로 리밸런싱한다. 비용 차감 후 평가금액이 목표 비중이 되도록 정확히 푼다.

    리밸런싱 후 총액을 V' 라 하면  V' = V - c * sum|t_i * V' - h_i|  를 만족해야 한다.
    c 가 작아 고정점 반복이 빠르게 수렴한다.
    반환값: (새 보유금액, 거래비용, 매매금액)
    """
    value = holdings.sum()
    new_value = value
    for _ in range(100):
        traded = np.abs(target * new_value - holdings).sum()
        candidate = value - cost_rate * traded
        if abs(candidate - new_value) < 1e-10 * max(value, 1.0):
            new_value = candidate
            break
        new_value = candidate
    traded = np.abs(target * new_value - holdings).sum()
    cost = value - new_value
    return target * new_value, cost, traded


def simulate(
    returns: pd.DataFrame,
    target_weights: dict[str, float],
    strategy: Strategy,
    initial_value: float = 100_000_000,
    cost_bp: float = 0.0,
) -> SimulationResult:
    """월별 수익률(소수, 0.01 = 1%)로 전략을 시뮬레이션한다.

    각 월: 월초 보유금액 → 해당 월 수익률 반영 → 월말에 리밸런싱 여부 판단.
    최초 매수 시에도 거래비용을 부과한다.
    """
    assets = list(target_weights)
    missing = [a for a in assets if a not in returns.columns]
    if missing:
        raise ValueError(f"수익률 데이터에 없는 자산: {missing}")
    target = np.array([target_weights[a] for a in assets], dtype=float)
    if np.any(target < 0):
        raise ValueError("목표 비중은 음수가 될 수 없습니다.")
    if not np.isclose(target.sum(), 1.0, atol=1e-6):
        raise ValueError(f"목표 비중 합계가 100%가 아닙니다: {target.sum():.4f}")

    rets = returns[assets].astype(float)
    if rets.isna().any().any():
        raise ValueError("수익률 데이터에 결측치가 있습니다.")
    if not np.isfinite(rets.to_numpy()).all():
        raise ValueError("수익률 데이터에 무한대 값이 있습니다. (가격 0 포함 여부를 확인하세요)")
    bad = rets.le(-1.0)
    if bad.any().any():
        date, asset = bad.stack().loc[lambda s: s].index[0]
        raise ValueError(
            f"{date} {asset}: 월 수익률 {rets.at[date, asset]:.2%} — -100% 이하 수익률은 "
            "자산 가치가 0 이하가 되므로 허용하지 않습니다. (소수 단위 입력인지 확인하세요: 0.01 = 1%)"
        )
    if not 0 <= cost_bp < 10_000:
        raise ValueError(f"거래비용은 0bp 이상 10,000bp 미만이어야 합니다: {cost_bp}")
    if initial_value <= 0:
        raise ValueError("초기 투자금은 0보다 커야 합니다.")
    cost_rate = cost_bp / 10_000

    # 최초 매수 (현금 → 목표 비중)
    value0 = float(initial_value)
    invest = value0 / (1 + cost_rate)
    holdings = target * invest
    total_cost = value0 - invest
    turnover = 0.0  # 최초 매수는 회전율에서 제외

    values, weight_rows, rebal_dates = [], [], []
    for i, (date, row) in enumerate(rets.iterrows()):
        holdings = holdings * (1 + row.to_numpy())
        value = holdings.sum()
        if value <= 0:
            raise ValueError(f"{date}: 포트폴리오 평가금액이 0 이하가 되었습니다.")
        weights = holdings / value
        if strategy.should_rebalance(i, weights, target):
            holdings, cost, traded = _rebalance(holdings, target, cost_rate)
            total_cost += cost
            turnover += traded / 2 / value
            rebal_dates.append(date)
        values.append(holdings.sum())
        weight_rows.append(holdings / holdings.sum())

    return SimulationResult(
        strategy=strategy.name,
        values=pd.Series(values, index=rets.index, name=strategy.name),
        weights=pd.DataFrame(weight_rows, index=rets.index, columns=assets),
        rebalance_dates=rebal_dates,
        turnover=turnover,
        total_cost=total_cost,
    )


def performance(result: SimulationResult, initial_value: float, risk_free: float = 0.0) -> dict:
    """성과 지표 계산. risk_free 는 연율(소수)."""
    values = result.values
    n_months = len(values)
    path = pd.concat([pd.Series([initial_value]), values.reset_index(drop=True)], ignore_index=True)
    monthly = path.pct_change().dropna()
    years = n_months / MONTHS_PER_YEAR
    total_return = values.iloc[-1] / initial_value - 1
    cagr = (values.iloc[-1] / initial_value) ** (1 / years) - 1 if years > 0 else np.nan
    vol = monthly.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) if n_months > 1 else np.nan
    rf_monthly = (1 + risk_free) ** (1 / MONTHS_PER_YEAR) - 1
    excess = monthly - rf_monthly
    sharpe = (
        excess.mean() / excess.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
        if n_months > 1 and excess.std(ddof=1) > 0
        else np.nan
    )
    drawdown = path / path.cummax() - 1
    return {
        "전략": result.strategy,
        "최종평가액": values.iloc[-1],
        "누적수익률": total_return,
        "연환산수익률(CAGR)": cagr,
        "연변동성": vol,
        "샤프지수": sharpe,
        "최대낙폭(MDD)": drawdown.min(),
        "리밸런싱횟수": len(result.rebalance_dates),
        "연평균회전율": result.turnover / years if years > 0 else np.nan,
        "누적거래비용": result.total_cost,
    }


DEFAULT_STRATEGIES = [
    Strategy("보유(리밸런싱 없음)", "none"),
    Strategy("월간", "periodic", period_months=1),
    Strategy("분기", "periodic", period_months=3),
    Strategy("반기", "periodic", period_months=6),
    Strategy("연간", "periodic", period_months=12),
    Strategy("밴드 ±5%p", "threshold", band=0.05),
    Strategy("밴드 ±10%p", "threshold", band=0.10),
]


def compare(
    returns: pd.DataFrame,
    target_weights: dict[str, float],
    strategies: list[Strategy] | None = None,
    initial_value: float = 100_000_000,
    cost_bp: float = 0.0,
    risk_free: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, SimulationResult]]:
    strategies = strategies or DEFAULT_STRATEGIES
    results = {s.name: simulate(returns, target_weights, s, initial_value, cost_bp) for s in strategies}
    table = pd.DataFrame([performance(r, initial_value, risk_free) for r in results.values()]).set_index("전략")
    return table, results


def sample_returns(n_years: int = 20, seed: int = 42) -> pd.DataFrame:
    """예시용 '가상' 월별 수익률. 실제 시장 데이터가 아니다.

    연 기대수익률/변동성 및 상관관계는 설명을 위한 대략적인 가정치다.
    """
    assets = ["국내주식", "해외주식", "국내채권", "금"]
    annual_mu = np.array([0.06, 0.08, 0.035, 0.05])
    annual_vol = np.array([0.20, 0.16, 0.05, 0.15])
    corr = np.array(
        [
            [1.00, 0.65, -0.10, 0.05],
            [0.65, 1.00, -0.15, 0.05],
            [-0.10, -0.15, 1.00, 0.20],
            [0.05, 0.05, 0.20, 1.00],
        ]
    )
    m_vol = annual_vol / np.sqrt(MONTHS_PER_YEAR)
    m_mu = annual_mu / MONTHS_PER_YEAR - 0.5 * m_vol**2
    cov = np.outer(m_vol, m_vol) * corr
    rng = np.random.default_rng(seed)
    log_r = rng.multivariate_normal(m_mu, cov, size=n_years * MONTHS_PER_YEAR)
    idx = pd.period_range("2000-01", periods=len(log_r), freq="M")
    return pd.DataFrame(np.expm1(log_r), index=idx, columns=assets)


def load_csv(path: str, kind: str) -> pd.DataFrame:
    """CSV 로드. 첫 열은 날짜. kind='prices' 면 가격을 월별 수익률로 변환."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if kind == "prices":
        if (df <= 0).any().any():
            raise ValueError("가격 데이터에 0 이하 값이 있습니다.")
        df = df.pct_change().iloc[1:]
    elif kind == "returns":
        pass
    else:
        raise ValueError("kind 는 'prices' 또는 'returns' 여야 합니다.")
    return df


def parse_weights(text: str) -> dict[str, float]:
    """'국내주식=30,해외주식=30,국내채권=40' 형식 (퍼센트 또는 소수)."""
    weights = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, _, val = part.partition("=")
        if not _:
            raise ValueError(f"비중 형식 오류: {part!r} (예: 국내주식=40)")
        weights[name.strip()] = float(val)
    total = sum(weights.values())
    if total > 1.5:  # 퍼센트로 입력한 것으로 간주
        weights = {k: v / 100 for k, v in weights.items()}
    return weights


def _format_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ["누적수익률", "연환산수익률(CAGR)", "연변동성", "최대낙폭(MDD)", "연평균회전율"]:
        out[col] = out[col].map(lambda x: f"{x:.2%}")
    out["샤프지수"] = out["샤프지수"].map(lambda x: f"{x:.2f}")
    for col in ["최종평가액", "누적거래비용"]:
        out[col] = out[col].map(lambda x: f"{x:,.0f}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="리밸런싱 전략 비교 시뮬레이션")
    p.add_argument("--csv", help="월별 데이터 CSV (첫 열 날짜). 생략 시 가상 예시 데이터 사용")
    p.add_argument("--kind", choices=["prices", "returns"], default="prices", help="CSV 값 종류 (기본: prices)")
    p.add_argument(
        "--weights",
        default="국내주식=20,해외주식=30,국내채권=40,금=10",
        help="목표 비중, 예: 국내주식=20,해외주식=30,국내채권=40,금=10",
    )
    p.add_argument("--initial", type=float, default=100_000_000, help="초기 투자금 (기본 1억원)")
    p.add_argument("--cost-bp", type=float, default=10.0, help="편도 거래비용 bp (기본 10bp = 0.10%%)")
    p.add_argument("--rf", type=float, default=0.03, help="무위험수익률 연율, 샤프지수용 (기본 0.03)")
    p.add_argument("--years", type=int, default=20, help="가상 데이터 기간(년)")
    p.add_argument("--seed", type=int, default=42, help="가상 데이터 시드")
    p.add_argument("--output", help="전략별 월말 평가금액을 저장할 CSV 경로")
    args = p.parse_args(argv)

    if args.csv:
        returns = load_csv(args.csv, args.kind)
        source = args.csv
    else:
        returns = sample_returns(args.years, args.seed)
        source = f"가상 예시 데이터 ({args.years}년, seed={args.seed}) — 실제 시장 데이터 아님"

    weights = parse_weights(args.weights)
    table, results = compare(returns, weights, initial_value=args.initial, cost_bp=args.cost_bp, risk_free=args.rf)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print(f"데이터: {source}")
    print(f"기간: {returns.index[0]} ~ {returns.index[-1]} ({len(returns)}개월)")
    print("목표 비중: " + ", ".join(f"{k} {v:.0%}" for k, v in weights.items()))
    print(f"초기 투자금: {args.initial:,.0f}원 / 거래비용: 편도 {args.cost_bp}bp / 무위험수익률: {args.rf:.2%}")
    print()
    print(_format_table(table).to_string())

    if args.output:
        pd.DataFrame({name: r.values for name, r in results.items()}).to_csv(args.output, encoding="utf-8-sig")
        print(f"\n평가금액 추이 저장: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as e:
        print(f"입력 오류: {e}", file=sys.stderr)
        sys.exit(2)
