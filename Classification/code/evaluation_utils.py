"""
문제 : 사이킷 런에서 제공하는 confusion matrix의 결과가 txt 파일에 저장되어있다. 이를 이차원 리스트 형태로 반환하여라.
입력 : Confusion Matrix의 txt 파일의 path
출력 : 이차원 리스트 반환
"""
def CMtoList(txt):
    f = open(txt,'r')

    # 2차원 배열
    matrix = []
    lines = f.readlines()

    for i in range(len(lines)-1):
        row = []
        line = lines[i]+lines[i+1]
        i += 1
        # 공백 및 불필요한 문자들 제거하여 각 숫자를 문자열로 저장 --> 나중에는 정규 표현식 써서 코드를 줄일 수 있도록!
        line = line.replace('[','')
        line = line.replace(']','')
        line = line.replace('\n','\t')
        line = line.replace('     ','\t')
        line = line.replace('    ','\t')
        line = line.replace('   ','\t')
        line = line.replace('  ','\t')
        line = line.replace(' ','\t')
        row = line.split('\t')
        for _ in range(20):
            if '' in row : row.remove('')
        #print(row)
        # 문자를 숫자로 변환
        row = list(map(int, row))
        matrix.append(row)

    return matrix
    


"""
문제 : 이차원리스트로 형성되어있는 Binary Classification Confusion Matrix에서 
False Positive Rate(fall-out value, false alarm rate)를 얻는다.
FPR = FP/FP+TN (실제로는 음성 클래스(0)인데 모델이 양성 클래스(1)로 오탐한 비율)
입력 : 2x2 이차원 리스트
입력 리스트 예시)
[[581512(TN),147(FP)]
 [45(FN),135762(TP)]]
출력 : Binary Classification의 FPR값
"""
def BinaryFPR(list):
    # 각각의 TP, FN, FP, TN 요소들을 선언
    tn = list[0][0]
    fp = list[0][1]
    fn = list[1][0]
    tp = list[1][1]

    # FPR 값 구하기
    fpr = fp/(fp+tn)
    return fpr


"""
문제 : 이차원리스트로 형성되어있는 Multiclass Classification Confusion Matrix에서 
각 class 별 precision, recall, f1 score을 얻는다.
입력 : NxN 이차원 리스트
출력 : class 별 precision, recall, f1 score, FPR
"""
import numpy as np
def MultiMetric(f, list):
    # print(list)
    print(len(list))
    
    list = np.array(list)
    # for i in range(len(list)):
    for i in range(len(list[:][0])-1):
        # print('class '+str(i),end=' : ')
        tp = list[i][i]

        tn = -tp
        for j in range(len(list[:][0])-1):
            tn += list[j][j]

        fp = list.sum(axis=0)[i]-tp
        sumval = list.sum()
        sumval = np.array(sumval)
        fn = sumval.sum()-tp-tn-fp
        
        print(tp, tn, fp, fn)
        acc = (tp+tn) / (tp+fp+tn+fn)
        precision = tp /(tp+fp)
        recall = tp / (tp+fn)
        f1score = 2 * precision * recall / (precision+recall)
        fpr = fp / (fp+tn)

        print(str(acc)+'\t'+str(f1score)+'\t'+str(recall)+'\t'+str(precision)+'\t'+str(fpr))
        f.write(str(acc)+'\t'+str(f1score)+'\t'+str(recall)+'\t'+str(precision)+'\t'+str(fpr)+'\n')

    f.close()
    
    