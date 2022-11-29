import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import time

from optimizer import Optimizer

# 파일 읽어오기
df_scfi = pd.read_csv("./data/scfi_monthly.csv", encoding='utf-8').dropna()
df_demand_factor = pd.read_csv("./data/load_factor.csv", encoding='utf-8')
# df_capacity = pd.read_csv("./data/capacity.csv", encoding='utf-8')

## 데이터 전처리
df_scfi['year'] = df_scfi['DATE'].apply(lambda x: '-'.join(x.replace('\xa0', '').split('-')[:1]))
df_scfi_yearly = df_scfi.groupby(by=['year']).mean()

## 14년치 데이터 전처리
df_scfi_yearly = df_scfi_yearly.drop(['SCFI'], axis=1)
# print(df_scfi_yearly)
# print(df_demand_factor)
capa_demand_list_base = np.array([154217, 86141, 42662, 36783, 71129])
df_demand_factor = df_demand_factor.set_index(['구분']).drop(['합계'], axis=1)

# print(df_demand_factor.values)
        
demand_tuned = np.zeros([14, 5])
for i in range(14):
    demand_tuned[i] = capa_demand_list_base * df_demand_factor.values[i]
# print(demand_tuned)
demand_tuned

class Tester():
    def __init__(self) -> None:
        self.opt = Optimizer()
        pass

    def optimize_one(self, freight_list, capa_demand_list):
        self.opt.optimizer_setting(freight_list, capa_demand_list)
        capa_supply_list = self.opt.get_optimized_deployment() # 결정변수 뽑기 -> 항로별 배치량 결정
        return capa_supply_list
    
    def cal_profit_one(self, freight_list, capa_demand_list, capa_supply_list):
        total = self.opt.cal_entire(freight_list, capa_demand_list, capa_supply_list) # -> 최종 수익 계산
        return total
    
    def optimize_many(self, freight_mat, capa_demand_mat):
        capa_supply_mat = np.zeros(capa_demand_mat.shape)

        for i in range(capa_demand_mat.shape[0]):
            capa_supply_mat[i] = self.optimize_one(freight_mat[i], capa_demand_mat[i])
        
        return capa_supply_mat
    
    def cal_profit_many(self, freight_mat, capa_demand_mat, capa_supply_mat):
        profit_list = np.zeros(capa_demand_mat.shape[0])

        for i in range(capa_demand_mat.shape[0]):
            profit_list[i] = self.cal_profit_one(freight_mat[i], capa_demand_mat[i], capa_supply_mat[i])
        
        return profit_list

tester = Tester()
capa_supply_mat = tester.optimize_many(df_scfi_yearly.values, demand_tuned)
print(capa_supply_mat)

real_capa_supply_mat = capa_supply_mat[:-1] # 2010년부터 적용한 실제 배치량
real_freight_mat_cont = df_scfi_yearly.values[:-1]
real_freight_mat_spot = df_scfi_yearly.values[1:]
real_capa_demand_mat = demand_tuned[1:]

real_freight_mat_list = [0] * 5
real_freight_mat_list[0] = real_freight_mat_cont * 0.0 + real_freight_mat_spot * 1.0
real_freight_mat_list[1] = real_freight_mat_cont * 0.25 + real_freight_mat_spot * 0.75
real_freight_mat_list[2] = real_freight_mat_cont * 0.5 + real_freight_mat_spot * 0.5
real_freight_mat_list[3] = real_freight_mat_cont * 0.75 + real_freight_mat_spot * 0.25
real_freight_mat_list[4] = real_freight_mat_cont * 1.0 + real_freight_mat_spot * 0.0

profit_mat_list = []
final_profit_mat = np.zeros([len(real_freight_mat_list), real_freight_mat_cont.shape[0]])
for i, fr_mat in enumerate(real_freight_mat_list):
    final_profit_mat[i] = tester.cal_profit_many(fr_mat, real_capa_demand_mat, real_capa_supply_mat)
    print(final_profit_mat)

final_profit_mat = final_profit_mat.T
print(final_profit_mat)
print(np.sum(final_profit_mat, axis=0))
print(np.mean(final_profit_mat, axis=0))
print(np.std(final_profit_mat, axis=0))

df_scfi_yearly.to_csv("./data/scfi_yearly")
pd.DataFrame(capa_supply_mat).to_csv("./result/capa_supply")
pd.DataFrame(demand_tuned).to_csv("./result/capa_demand")


# profit_list_cont = tester.cal_profit_many(real_freight_mat_cont, real_capa_demand_mat, real_capa_supply_mat)
# profit_list_spot = tester.cal_profit_many(real_freight_mat_spot, real_capa_demand_mat, real_capa_supply_mat)

# print("contract!")
# print(profit_list_cont)
# print(np.mean(profit_list_cont))
# print(np.std(profit_list_cont))
# print()
# print("spot!")
# print(profit_list_spot)
# print(np.mean(profit_list_spot))
# print(np.std(profit_list_spot))


