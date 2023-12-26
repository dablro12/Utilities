#!/usr/bin/env python
import subprocess as sp
import parmap 
import multiprocessing as mp # 내장 package
import numpy as np 
try:
    import tqdm 
except ImportError:
    sp.run(['pip' 'install' 'tqdm'])
    import tqdm
    
def main(): 
    """
    멀티프로세싱할 메인 함수 정의
    """
    data = 1 
    return data 

if __name__ == "__main__":
    num_cores = mp.cpu_count()
    data = """ 데이터 정의 """
    splited_data = np.array_split(data, num_cores) # 데이터를 cpu코어에 맞게 분할
    splited_data = [x.tolist() for x in splited_data] #-> numpy 결과를 모두 각 코어인덱스별 리스트로 반환
    
    ## parmap 으로 cpu multiprocessing 수행
    """
    arguments
        func : 수행할 함수
        data : 코어별로 분할된 입력 데이터
        pm_pbar = tqdm 베이스 수행 현황 공유
        pm_processes = cpu core 수     
    """
    ## main 함수 인자가 여러 개이면 starmap을 이용해야됨 
    result = parmap.map(func = main, data = splited_data, pm_pbar = True, pm_processes = num_cores)
    # result = parmap.starmap(main, zip(splited_data, x2, x3, ...), pm_pbar = True, pm_processes = num_cores)
    
    
    ## parmap은 각 코어별로 수행한 결과를 리스트로 반환
    if isinstance(result, list):
        result = sum(result, []) # result에 각 수행 결과를 리스트로 반환 받기 
    