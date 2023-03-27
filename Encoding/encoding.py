import pandas as pd
import encoding_utils as eu

files =[
    "CICItoCICI",
    "CICItoUNSW",
    "UNSWtoCICI",
    "UNSWtoUNSW",
    "ConcatedCICI",
    "ConcatedUNSW"
]

# Pipeline for Encoding
# clean columns are already done
def pipeline(filename):
    data = eu.getData(filename)
    # data = cleanColumns(filename, data)
    x_data = eu.getX(data)
    nf, cf = eu.devideFeatures(x_data)
    eu.printFeatures(data, nf, cf)
    x_data, n_data, c_data = eu.xEncoding(x_data, nf, cf)
    y_data = eu.yEncoding(data)
    eu.returnData(filename, x_data, y_data)
    
# Encoding All Files
for file in files:
    pipeline(file)