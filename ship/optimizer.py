import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import time

from mip import *

# 파일 읽어오기
df_scfi = pd.read_csv("./data/scfi_monthly.csv", encoding='utf-8').dropna()
df_demand_factor = pd.read_csv("./data/load_factor.csv", encoding='utf-8')
df_capacity = pd.read_csv("./data/capacity.csv", encoding='utf-8')

## 데이터 전처리
df_scfi['year'] = df_scfi['DATE'].apply(lambda x: '-'.join(x.replace('\xa0', '').split('-')[:1]))
df_scfi_yearly = df_scfi.groupby(by=['year']).mean()


## 숫자로 변환
def numeric(x):
    if type(x) == str:
        return float(x.replace(' ','').replace(',', ''))
    else:
        return x


class Optimizer():
    def __init__(self):
        # 기본 정보
        self.load_rate_list = np.array([0.416, 0.257, 0.41, 0.596, 0.284]) # 적취율
        self.cycle_time_list = np.array([58.69, 105, 37.11, 23.77, 54.22]) # 사이클타임
        self.freight_list = np.array([0, 0, 0, 0, 0]) # 운임: USD / TEU
        # self.freight_list = np.array([1, 1, 1, 1, 1]) 

        # 수요 물동량, 실제 할당량
        self.capa_demand_list = np.array([0, 0, 0, 0, 0]) # TEU / YR
        self.capa_supply_list = np.array([0, 0, 0, 0, 0]) # TEU / YR

        self.cost_fixed_yearly = 539.08 * 1e6 # $
        self.cost_fixed_yearly_list = np.array([282.15, 48.05, 47.04, 56.43, 105.41]) * 1e6 # $
        self.cost_var_ratio_yearly_list = np.array([0.181, 0.177, 0.181, 0.187, 0.177]) # 매출액 중 운항비 비율
        self.not_cost_ratio = np.array([1] * 5) - self.cost_var_ratio_yearly_list
        # print(self.not_cost_ratio)

        self.set_capacity_data()
    
    def set_capacity_data(self):
        self.df_capacity = pd.read_csv("./data/capacity.csv", encoding='utf-8')
        self.df_capacity.columns=['name', 'TEU']
        self.df_capacity['TEU'] = self.df_capacity['TEU'].apply(lambda x: numeric(x))
    
    def cal_freight_yearly(self):
        # 사이클타임 고려한 연간 TEU 당 가격 계산
        self.freight_yearly_list = np.array([365] * 5) / self.cycle_time_list * self.freight_list # USD / TEU / YR
        # print(f"freight_yearly_list: {self.freight_yearly_list}")
        return self.freight_yearly_list
    
    def cal_real_demand_yearly(self):
        # 적취율 고려한 실제 물동량, 공급량
        self.real_demand_list = self.capa_demand_list * self.load_rate_list # TEU / YR
        self.real_supply_list = self.capa_supply_list * self.load_rate_list # TEU / YR
        self.real_demand_list = np.minimum(self.real_demand_list, self.real_supply_list)
        print(f"real_demand_list: {self.real_demand_list}")
        print(f"real_supply_list: {self.real_supply_list}")
    
    def cal_sum_cost(self):
        # 공급량 고려한 연간 총 비용
        self.cost_var_yearly_list = self.real_supply_list * self.freight_yearly_list * self.cost_var_ratio_yearly_list
        # print(f"cost_var_yearly_list: {self.cost_var_yearly_list}")
        self.sum_cost = np.sum(self.cost_fixed_yearly_list) + np.sum(self.cost_var_yearly_list)
        # print(np.sum(self.cost_fixed_yearly_list))
        # print(np.sum(self.cost_var_yearly_list))
        print("sum_cost: ", self.sum_cost / 1e6)
    
    def cal_sum_sales(self):
        # 실제 수요량 고려한 연간 총 매출
        # print("real demand list: ", self.real_demand_list)
        # print("fr list: ", self.freight_yearly_list)
        self.sum_sales = np.sum(self.real_demand_list * self.freight_yearly_list)
        print(f"sum_sales: {self.sum_sales / 1e6}")

    def cal_total_profit(self):
        self.total_profit = self.sum_sales - self.sum_cost
        print(f"total_profit: {self.total_profit / 1e6}")
        return self.total_profit / 1e6
    
    def cal_entire(self, freight_list, capa_demand_list, capa_supply_list): # input: np.array, [5]
        self.capa_demand_list = capa_demand_list
        self.capa_supply_list = capa_supply_list
        self.cal_freight_yearly()
        self.freight_list = freight_list
        self.cal_real_demand_yearly()
        self.cal_sum_cost()
        self.cal_sum_sales()
        total_M = self.cal_total_profit()
        return total_M

        # info.freight_list = np.array([975.16, 883.91, 377.40, 224.59, 1092.42])
        # info.capa_demand_list = np.array([154217, 86141, 42662, 36783, 71129])
        # info.capa_supply_list = np.array([171352, 95712, 47402, 40870, 79032])
    
    def optimizer_setting(self, freight_list, capa_demand_list):
        self.freight_list = freight_list
        self.capa_demand_list = capa_demand_list

        self.capacity = list(self.df_capacity['TEU'])
        self.demand = list(capa_demand_list)
        self.fr = list(freight_list)
        self.load_ratio = list(self.load_rate_list)
        self.cost_ratio = list(self.cost_var_ratio_yearly_list)

        self.model = Model()

        # number of vessels
        self.n = 51
        # number of lines
        self.m = 5

        # set variables
        self.x = [[self.model.add_var('x({},{})'.format(i, j), var_type=BINARY) for j in range(self.m)]
                    for i in range(self.n)]

        self.model.objective = minimize(xsum(self.fr[j] * self.load_ratio[j] * self.cost_ratio[j] * self.x[i][j] for i in range(self.n) for j in range(self.m)))

        # Constraints
        # one ship only one line
        for i in range(self.n):
            self.model += xsum(self.x[i][j] for j in range(self.m)) <= 1, 'row({})'.format(i)

        for j in range(self.m):
            self.model += xsum(self.capacity[i] * self.x[i][j] for i in range(self.n)) >= self.demand[j], 'row({})'.format(i)

    def get_optimized_deployment(self):
        self.model.max_gap = 0.05
        status = self.model.optimize(max_seconds=5)
        if status == OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(self.model.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(self.model.objective_value, self.model.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(self.model.objective_bound))
        elif status == OptimizationStatus.INFEASIBLE:
            print('no feasible solution found, lower bound is: {}'.format(self.model.objective_bound))
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            print('solution:')
            for v in self.model.vars:
                if abs(v.x) > 1e-6: # only printing non-zeros
                    # print('{} : {}'.format(v.name, v.x))
                    pass
        
        count = 0
        matrix = np.zeros([self.n, self.m])
        for i in range(self.n):
            for j in range(self.m):
                if self.model.vars[count].x > 1e-6:
                    matrix[i][j] = 1
                count += 1

        np_capacity = np.array(self.capacity)
        # print(matrix)
        self.capa_supply = np.matmul(np_capacity, matrix)
        print(self.capa_supply)
        self.capa_supply_list = self.capa_supply

        return self.capa_supply_list




# model.optimize()

# model.max_gap = 0.05
# status = model.optimize(max_seconds=5)
# if status == OptimizationStatus.OPTIMAL:
#     print('optimal solution cost {} found'.format(model.objective_value))
# elif status == OptimizationStatus.FEASIBLE:
#     print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
# elif status == OptimizationStatus.NO_SOLUTION_FOUND:
#     print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
# elif status == OptimizationStatus.INFEASIBLE:
#     print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
# if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
#     print('solution:')
#     for v in model.vars:
#        if abs(v.x) > 1e-6: # only printing non-zeros
#           print('{} : {}'.format(v.name, v.x))
#           pass

if __name__ == "__main__":
    info = Optimizer()
    freight_list = np.array([975.16, 883.91, 377.40, 224.59, 1092.42])
    capa_demand_list = np.array([154217, 86141, 42662, 36783, 71129])
    info.optimizer_setting(freight_list, capa_demand_list)
    capa_supply_list = info.get_optimized_deployment()
    info.cal_entire(freight_list, capa_demand_list, capa_supply_list)


