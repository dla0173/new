# new

## 리밸런싱 시뮬레이터 (`rebalance/`)

월별 수익률(또는 가격) 데이터로 리밸런싱 전략을 비교합니다.

- 보유(리밸런싱 없음), 월간/분기/반기/연간 정기 리밸런싱, 밴드(±5%p, ±10%p) 리밸런싱
- 거래비용(편도 bp) 반영: 비용 차감 후에도 정확히 목표 비중이 되도록 계산
- 지표: 최종평가액, 누적수익률, CAGR, 연변동성, 샤프지수, MDD, 리밸런싱 횟수, 연평균 회전율, 누적 거래비용

### 실행

```bash
# 가상 예시 데이터 (실제 시장 데이터 아님)
python3 -m rebalance.simulate

# 비중·비용 변경
python3 -m rebalance.simulate --weights "국내주식=30,해외주식=30,국내채권=40" --cost-bp 15

# 실제 데이터: 첫 열이 날짜, 나머지 열이 자산별 월말 가격인 CSV
python3 -m rebalance.simulate --csv prices.csv --kind prices \
  --weights "KOSPI200=30,SP500=30,국고채10년=40" --output values.csv

# 테스트
python3 -m unittest rebalance.test_simulate -v
```

### 한계

세금(양도세·배당세), 슬리피지, 최소 거래단위, 환율·환헤지 비용, 현금흐름(추가 납입·인출)은 반영하지 않습니다.
고객 안내 자료로 쓸 때는 실제 지수 데이터를 사용하고, 과거 성과가 미래 수익을 보장하지 않는다는 점을 함께 고지하세요.
