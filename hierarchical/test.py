
import numpy as np

x=np.arange(10)
y=np.arange(5,15)

print(x[np.where(np.isin(x,y)==True)]) # x랑 y랑 동시에 있는 값 반환 
