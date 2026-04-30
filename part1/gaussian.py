import warnings

def gaussian_eliminate(A, b):
    # Tao ma tran mo rong
    M=[]
    num_rows=int(len(A))
    num_cols=int(len(A[0]))

    for i in range(num_rows):
        M.append(A[i] + [b[i]])
    s=0
    epsilon = 1e-12
    pivot_row=0
    pivot_col=0
    while pivot_row<num_rows and pivot_col<num_cols:
        p=pivot_row
        max_val=abs(M[pivot_row][pivot_col])
        for i in range(pivot_row+1,num_rows):
            if abs(M[i][pivot_col])>max_val:
                max_val=abs(M[i][pivot_col])
                p=i
        if max_val==0:
            pivot_col+=1
            continue
        if max_val<epsilon:
            warnings.warn(f"Pivot tai cot {pivot_col} gan bang 0 ({max_val:.2e}). He co the ill-conditioned.")
        if p!=pivot_row:
            M[p],M[pivot_row]=M[pivot_row],M[p]
            s+=1
        for i in range(pivot_row+1,num_rows):
            l=M[i][pivot_col]/M[pivot_row][pivot_col]
            for j in range(pivot_col,num_cols+1):
                M[i][j]-=l*M[pivot_row][j]
        pivot_row+=1
        pivot_col+=1
        
    U = [row[:num_cols] for row in M]
    c = [row[num_cols] for row in M]
    x = back_substitution(U, c)
    return M,x,s

def back_substitution(U, c):
    num_rows=len(U)
    num_cols=len(U[0])
    epsilon=1e-12
    pivot_indicies=[]
    for row in range(num_rows):
        has_pivot=False
        for col in range(num_cols):
            if abs(U[row][col])!=0:
                pivot_indicies.append((row,col))
                has_pivot=True
                break
        if not has_pivot and abs(c[row])!=0:
            return 'Vo nghiem'
    
    pivot_cols=[p[1] for p  in pivot_indicies]
    free_cols=[i for i in range(num_cols) if i not in pivot_cols]
    x=[None]*num_cols
    for j in range(num_cols):
        if j not in pivot_cols:
            x[j]=f'x{j+1}'
    for i,j in reversed(pivot_indicies):
        num_val=c[i]
        not_num_val=[]
        for k in range(j+1,num_cols):
            if abs(U[i][j])!=0:
                if k in free_cols:
                    sign='-' if U[i][k]>0 else '+'
                    not_num_val.append(f'{sign}{abs(U[i][k])}*x{k+1}')
                elif isinstance(x[k],(int,float)):
                    num_val-=U[i][k]*x[k]
        if len(not_num_val)==0:
            x[j]=num_val/U[i][j]
        else:
            res=f'{num_val}/{U[i][j]}'
            for item in not_num_val:
                if item[0]=='+':
                    res+=f' + {item[1:]}'
                else:
                    res+=f' - {item[1:]}'
            res='('+res+')'+f'/{U[i][j]}'
            x[j]=res
    return x
                       