# Implementation
## 34E. Collisions
思路：算出所有质点两两之间的碰撞时间，每当两个质点碰撞时（所有球碰撞时间中的最小值），更新他们俩的速度和包括他俩在内周围四个球的碰撞时间。
```python
[n,t] = map(int,input().split())
data = sorted([list(map(int,input().split())) +[_] for _ in range(n)])
ct = [0]*(n-1)
time = 0
def updater(a,b):
    global time
    for i in range(a,b):
        if -1 < i < n-1:
            dx = (data[i+1][0]-data[i][0])
            dv = (data[i][1]-data[i+1][1])
            if dv > 0:
                ct[i] = dx/dv + time
            else:
                ct[i] = 10**12
updater(0,n-1)
while time < t:
    t_next = min(min(ct),t)
    for i in range(n):
        data[i][0] += data[i][1]*(t_next-time)
    time = t_next
    for i in range(n-1):
        if ct[i] == t_next:
            m1, m2 = data[i][2], data[i+1][2]
            v1, v2 = data[i][1], data[i+1][1]
            new_v1 = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            new_v2 = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
            data[i][1], data[i+1][1] = new_v1, new_v2
            updater(i-1,i+2)
data.sort(key = lambda x:x[3])
for i in range(n):
    print(f"{data[i][0]:.9f}")
```
# Math Problem（Observation）
## 1327A. Sum of Odd Integers
显然 $i$ 和 $t$ 奇偶性相同，再限制 $i$ 不能超过 $t$ 个不同奇数所能表示的最小值 $t^{2}$ 即可。
```python
n = int(input())
for _ in range(n):
	[i,t] = map(int,input().split())
	if (i+t)%2 == 1 or i < t**2:
		print("NO")
	else:
		print("YES")
```

## 1352C. K-th Not Divisible by n
为什么tag上又有binary search啊
给出一个不用分类讨论的做法。
我们可以考虑在前 $k$ 个数中间漏掉了多少数，把这些数的个数加上就得到了答案。
已知每 $n-1$ 个数之后就会有一个漏掉的数，总的完整组数（含有 $n-1$ 个数和一个被整除的数）是
$\lceil \dfrac{k}{n-1} -1\rceil$ （不论整除与否），于是可轻松得到答案。
```python
from math import ceil
for _ in range(int(input())):
    [n,k] = map(int,input().split())
    print((ceil(k/(n-1))-1)+k)
```
## M12558:岛屿周长
这题并不需要dfs，注意到每一个为 $1$ 的点对周长的贡献为 $4-c$ ，其中 $c$ 是这个点连接到 $1$ 的数量。遍历一下即可。
```python
[n,m] = map(int,input().split())
land = []
for i in range(n):
    land.append(list(map(int,input().split())))
l = 0
for i in range(n):
    for j in range(m):
        con = 0
        if land[i][j]:
            for a,b in [(0,1),(0,-1),(1,0),(-1,0)]:
                if 0<=i+a<n and 0<=j+b<m:
                    con += land[i+a][j+b]
            l += 4-con
print(l)
```
## M01852:Ants
trick：由于两只蚂蚁碰撞后交换了速度，我们可以认为它们也交换了掉下去的时间。于是事实上我们只需要将每只蚂蚁掉下去的时间作适当互换（这不会改变最后一只蚂蚁掉下去的时间），就可以忽略蚂蚁之间的碰撞。所以问题相当于问：无碰撞情况下所有蚂蚁都走下棍子的所花时间的最小值和最大值。
最小值情况，所有蚂蚁都朝离自己近的一端走去。
最大值情况，所有蚂蚁都朝离自己远的一端走去。
则时间易求。
```python
n = int(input())
for _ in range(n):
    [l,m] = map(int,input().split())
    nums = list(map(int,input().split()))
    distance = [min(l-nums[i],nums[i]) for i in range(m)]
    print(max(distance),l-min(distance))
```

## 2084B. MIN = GCD
解：假设 $a_{m}$ 是 $\{a_{i}\}$ 中的最小值，那么一定有
$$
\min([a_{1},a_{2},…,a_{i}])=\gcd([a_i+1,a_{i}+2,…,a_{n}]) = a_{m}
$$
证明：用反证法，如果 $\min([a_{1},a_{2},…,a_{i}]) = a_{j}\neq a_{m}$，那么 $\gcd([a_i+1,a_{i}+2,…,a_{n}]) \leq \min([a_i+1,a_{i}+2,…,a_{n}]) = a_{m} < \min([a_{1},a_{2},…,a_{i}])$，矛盾。
我们只需要判断剩下所有数中有没有数满足条件即可。
1. $\exists \;a_{j} = a_{m},\;j \neq m$ 
2. $\gcd(a_{i},a_{j}) = a_{m},i \neq j$ 
我们只需挨个更新gcd即可。
```python
from math import gcd
for _ in range(int(input())):
    n = int(input())
    ns = list(map(int,input().split()))
    nm = min(ns)
    ns.remove(nm)
    y = 0
    g = 0
    for j in range(n-1):
        if ns[j] % nm == 0:
            if g == 0:
                g = ns[j]
            else:
                g = gcd(g,ns[j])
        if ns[j] == nm or g == nm:
            y = 1
    print("YES" if y else "NO")
```

## 2152B. Catching the Krug
（仍然不知道为什么tags有binary search）
2152还是太难了，20min快速AC2之后C题写了30min喜提TLE，然后就睡觉去了。
容易看出由于Krug和Doran每次必定在一个方向上距离不减，所以Krug到达角落之前Doran是追不到的（除开两者在同一条水平竖直线的情况，此时存活时间显然，因为Doran可沿对角线移动，所以Krug在垂直于追击方向的移动无意义）。所以问题变成Doran多久能到达追击方向对应的角落，此时间就是Doran到两边距离的最大值。
```python
g = int(input())
for _ in range(g):
	n,rk,ck,rd,cd = map(int,input().split())
	print(max((n-rd)*(rk>rd)+rd*(rk<rd),(n-cd)*(ck>cd)+cd*(ck<cd))
```

## 2152C. Triple Removal
考试的时候居然用brute force，自然是TLE了。
首先判断能否实现：要求有 $3n$ 个 0 和 $3m$ 个 1，这可以用前缀和快速判断。
接着，我们证明，除非任意一个数两侧都是与他不同的数（此时cost为 $n+m+1$ ），否则cost为 $n+m$ 。
$Proof:$
首先，每一步的cost不可能 $\geq 3$ ，这很显然。
接着，我们考虑“任意一个数两侧都是与他不同的数”不成立的情况。
分以下情况讨论：（设 $a,b$ 为不同的数）
$1.$ 出现 $a\;b\;b\;a$ 的情况，此时消去两个相邻的 $b$ 和其他任意一个 $b$ 即可。
$2.$ 出现 $a\;b\;b\;b\;a$ 的情况，此时消去三个相邻的 $b$ 即可。
$3.$ 出现 $a\;b\;b\;b\;b\;a$ 的情况，此时消去四个相邻的 $b$ 中任意两个相邻的 $b$ 和其他任意一个 $b$ 即可。
$4.$ 两个 $a$ 间 $b$ 的数量多于 $4$ 个，此时通过消去三的倍数个 $b$ 一定可以转化为 $1.2.3.$ 中的一个情况。
这些步骤的cost都是 $1$ 。
在上述操作后，“任意一个数两侧都是与他不同的数”仍然不成立（都有至少两个 $a$ 或 $b$ 相邻），所以可以递推得到总 cost 为 $n+m$
至于”任意一个数两侧都是与他不同的数“成立的情况，只需消去任意一组cost为 $2$ 的数，就转化为了“任意一个数两侧都是与他不同的数”不成立的情况，所以总 cost 为 $n+m+1$ 。
那么如何判断呢？我们可以再次利用前缀和，若第 $i$ 项和第 $i+1$ 项相同就 $+1$，否则不加。如果开头和结尾的值相同，说明满足条件。
```python
for _ in range(int(input())):
    [m,q] = map(int,input().split())
    l = list(map(int,input().split()))
    p = [0]*(m+1)
    o = [0]*m
    for i in range(1,m+1):
        p[i] = p[i-1] + l[i-1]
        if i != m:
            o[i] = o[i-1]+(l[i-1] == l[i])
    for __ in range(q):
        [s,e] = map(int,input().split())
        d = e-s+1
        if (p[e]-p[s-1])%3 != 0 or d %3 != 0:
            print(-1)
        else:
            print(d//3+1 if o[e-1]-o[s]==0 else d//3)
```

## 2152D. Division Versus Addition
这个题和C题的思路可以说是一模一样啊，然后CF定的难度就从1400变成1700了。
将所有数用二进制表示可以有效帮助我们理解。
首先分析只含有一个数 $n$ 的情况，容易看出需要的步数是

$$
\mathrm{cost} = \lceil \log_{2}n \rceil 
$$

当含有多个数的时候，双方要抢的是形式为 $2^k+1(k>0)$ 的数，因为如果Poby先对这个数整除以 $2$ 那么就只需要 $n$ 步就可以把它降到 $1$ ，而如果Rekkles先对这个数加 $1$ ，那么就需要 $n+1$ 步。所以Poby能够在单个数步数之和的基础上减少一些步数，其具体的值为

$$
\lceil \dfrac{N(2^k+1)}{2} \rceil
$$

在对于同一个片段作多次query时，这两者都可以通过前缀和极快的计算完成。
```python
from math import ceil,log2,floor
for _ in range(int(input())):
    [n,q] = map(int,input().split())
    nums = list(map(int,input().split()))
    l = [0]*(n+1)
    save = [0]*(n+1)
    for i in range(1,n+1):
        l[i] = l[i-1] + ceil(log2(nums[i-1]))
        save[i]  = save[i-1] + int(ceil(log2(nums[i-1]-1)) == floor(log2(nums[i-1]-1)) and nums[i-1]>2)
    for __ in range(q):
        [s,e] = map(int,input().split())
        print(l[e]-l[s-1]-ceil((save[e]-save[s-1])/2))
```

## 2131C. Make it Equal（Fake）
完全错误的做法，但是AC，正确的做法应该是比对每一项。
```python
import sys
input = sys.stdin.readline
for _ in range(int(input())):
    [n,k] = map(int,input().split())
    s,n = 0,0
    for i in input().split():
        ik = int(i)%k
        s += min(k-ik,ik)
        n += ik == 0
    for i in input().split():
        ik = int(i)%k
        s -= min(k-ik,ik)
        n -= ik == 0
    if s == 0 and n == 0:
        sys.stdout.write("yes\n")
    else:
        sys.stdout.write("no\n")
```

## M19948:因材施教
考虑排序后每个数单独分为一组（相当于中间画 $n-1$ 条竖线将每个数分隔开），现在我们要拆除 $n-m$ 条竖线，我们发现，拆除一条竖线后，总“差异值”的增加量就是左右两个数的差异。于是我们只需要取前 $n-m$ 组差异最小的数对拆除即可。
```python
[n,m] = map(int,input().split())
nums = sorted(list(map(int,input().split())))
cost = sorted([nums[i+1]-nums[i] for i in range(n-1)])
print(sum(cost[:n-m]))
```

## M29853:小蓝与小桥
我们考虑把所有pair之间的差写成一个 $n\times n$ 的矩阵，比如对于题目样例，写作：
```python
1 19
8 10
```
那么小蓝与小桥删题的过程可以看作交替删去最大数所在行和最小数所在列，直到只剩一个数。那么这个数必定是所有行中最大数中的最小数，比如样例中两行的最大值分别是 $19$ 和 $10$ ，那么剩下的数就是其中的最小值 $10$ 
```python
n = int(input())
alist = list(map(int,input().split()))
blist = sorted(list(map(int,input().split())))
ans = float("inf")
for a in alist:
    ans = min(ans,max(abs(a-blist[0]),abs(a-blist[-1])))
print(ans)
```
## 1366D. Two Divisors
首先学习了欧拉筛法，其优越性在于不会重复遍历合数。
接着就是这个题重要的数论知识，首先，任意可以表示成 $p^k$ 形式的数都不符合条件（ $p$ 为素数），对于其他数 $a_{i}$ ，设其最小质因数为 $p$ ，$a_{i} = p^k \cdot q$ 且 $\gcd(p^k,q) = 1$ 。那么 $p^k$ 和 $q$ 就是满足条件的数。
证明：

$$
\gcd(p^k+q,p^k q) = \gcd(p^k+q,\gcd(p^k+q,p^k)⋅q) = \gcd(p^k+q,q)= 1
$$

于是我们只需找出每个数的最小质因数 $p$ 即可。
```python
import array
min_prime = [0] * (10**7 + 1)
for i in range(2, 10**7 + 1):
    if min_prime[i] == 0:
        min_prime[i] = i
        for j in range(i * i, 10**7 + 1, i):
            if min_prime[j] == 0:
                min_prime[j] = i
def tdv(x):
    x0 = x
    npf = min_prime[x0]
    while x0 % npf == 0:
        x0 //= npf
    if x0 != 1:
        b.append(npf)
        return x0
    b.append(-1)
    return -1
b = []
n = int(input())
nl = array.array("i",list(map(int,input().split())))
for i in range(n):
    t = tdv(nl[i])
    if i == n-1:
        print(t)
    else:
        print(t,end = " ")
print(*b)
```
## T09267:核电站
递推思想，首先 $n<m$ 时肯定有 $2^n$ 中方法，当 $n\geq m$ 时考虑添加第 $n$ 个坑，此时放法种类数为 $w[n]$，若此坑不放，则有 $w[n-1]$ 中放法，如果要放，那么从 $w[n-1]$ 种放法中找到不合法的，由于放第 $n$ 个之前合法，那么要求倒数第 $m$ 个不放，后面 $m-1$ 个坑都要放，剩下 $n-m-1$ 个坑随便放。最终递推式为

$$
w[n] = 2w[n-1]-w[n-m-1]
$$

```python
[n,m] = map(int,input().split())
w = []
if n < m:
    print(1<<n)
else:
    for i in range(n+1):
        if i <= m:
            w.append((1<<i)-i//m)
        else:
            w.append(2*w[i-1]-w[i-m-1])
    print(w[-1])
    ```

# Data Structure
## M03704:扩号匹配问题
练习一下双向队列deque，这个题从左往右找括号就可以

```python
from collections import deque
import sys
data = sys.stdin.read().splitlines()
for s in data:
    n = 0
    index = deque()
    bra = deque()
    for i in range(len(s)):
        if s[i] == "(":
            n += 1
            bra.append("(")
            index.append(i)
        elif s[i] == ")":
            if n>0:
                bra.pop()
                index.pop()
                n -= 1
            else:
                bra.append(")")
                index.append(i)
    o = list(" "*len(s))
    for i in range(len(bra)):
        if bra[i] == "(":
            o[index[i]] = "$"
        elif bra[i] == ")":
            o[index[i]] = "?"
    print(s)
    print("".join(o))
```

# Binary Search
## M04135:月度开销
做的依托，之前有一点思路，就是从所有月份中最大值开始一点一点加费用，直到加到只有m个fajo月。但是肯定TLE就没做，后面根据群里提示可以二分查找，然后边界条件就做的很粗糙，最后还是先用二分法快速逼近到一个近似解，然后再慢慢达到正确解。中间也是踩了很多坑。
1. 最优解fajo月的数量可以少于 $m$ 。
2. $m$ 个fajo月的方案也可能有多种划分方式，不一定最优。
```python
[n,m] = map(int,input().split())
num = [int(input()) for _ in range(n)]
min_s = max(num)
max_s = sum(num)
ss = []
def f():
    global cs,s,ss
    ss = []
    s = 0
    for i in range(n):
        if s + num[i] > cs:
            ss.append(s)
            s = 0
        s += num[i]
    ss.append(s)
while max_s > min_s + 1:
    cs = (min_s+max_s)//2
    f()
    if len(ss) > m:
        min_s = cs
    elif len(ss) < m:
        max_s = cs
    else:
        cs = min(cs,max(ss))
        break
if len(ss) > m:
    while 1:
        cs += 1
        s = 0
        f()
        if len(ss) <= m:
            cs = min(cs,max(ss))
            break
if len(ss) <= m:
    while 1:
        cs -= 1
        f()
        if len(ss) > m or cs < max(num):
            cs += 1
            break
print(cs)
```

# 并查集
并查集实质上就是一个包含所有节点的父节点的列表，其中包含了查找（find）和合并（union）方法。
如果直接用字典表示从属关系，那么在最坏情况下，其会退化成链表，查找时效率低下，所以我们在find方法中添加了路径压缩功能，即把每一个子节点都直接与同一个父节点相连，此时查找效率更高。
下面举一个例子：
如果输入为
```
1 2
2 3
3 4
```
那么直接用散列表表示从属关系，就会是
$$
1\to 2\to 3\to 4
$$
其效率低下，而利用并查集的路径压缩，则从属关系变为
$$
1\to 2,3,4
$$
查找效率更高。
注意由于我们只在合并时调用find方法，所以可能出现父节点并没有完全归并的情况，所以最后还应对每一个元素再次调用find方法。
比如说下列一维上的并查集问题：
## M18250:冰阔落 I
```python
import sys
data = sys.stdin.read().splitlines()
index = 0
case = 1
class Unionfind():
    def __init__(self,n):
        self.count = n
        self.p = list(range(n))
    def find(self,x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        if self.find(x) != self.find(y):
            self.p[self.find(y)] = self.find(x)
            self.count -= 1
    def connected(self,x,y):
        if self.find(x) == self.find(y):
            print("Yes")
        else:
            self.union(x,y)
            print("No")
    def update(self):
        for _ in range(n):
            self.p[_] = self.find(_)
while index < len(data):
    [n,m] = list(map(int,data[index].split()))
    uf = Unionfind(n)
    for i in range(m):
        index += 1
        [x,y] = list(map(int,data[index].split()))
        uf.connected(x-1,y-1)
    uf.update()
    print(uf.count)
    print(" ".join(map(str,sorted(set([uf.p[i] + 1 for i in range(len(uf.p))])))))
    index += 1
```

接下来我们就可以解决二维上的并查集问题
## M02815:城堡问题
由于列表是不能当下标的，所以用 $100i+j$ 来表征。
```python
class Unionfind():
    def __init__(self,n,m):
        self.count = n*m
        self.p = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                self.p[i][j] = i*100+j
    def find(self,x,y):
        if self.p[x][y] != x*100+y:
            self.p[x][y] = self.find(self.p[x][y]//100,self.p[x][y]%100)
        return self.p[x][y]
    def union(self,x1,y1,x2,y2):
        rt1,rt2 = self.find(x1,y1),self.find(x2,y2)
        if rt1 != rt2:
            self.p[rt2//100][rt2%100] = rt1
            self.count -= 1
n = int(input())
m = int(input())
state = [[0] * m for _ in range(n)]
for i in range(n):
    data = list(map(int,input().split()))
    for j in range(m):
        state[i][j] = bin(data[j])[2:].zfill(4)
uf = Unionfind(n,m)
for i in range(n):
    for j in range(m):
        if state[i][j][0] == "0":
            uf.union(i,j,i+1,j)
        if state[i][j][1] == "0":
            uf.union(i,j,i,j+1)
print(uf.count)
size = {}
for i in range(n):
    for j in range(m):
        root = uf.find(i, j)
        if root in size:
            size[root] += 1
        else:
            size[root] = 1
print(max(size.values()))
```

## 01258:Agri-Net
MST问题，并查集加贪心，将每两家农民之间道路的开销排序，从小到大取，如果两个节点已经连通就跳过。
```python
import sys
class UnionFind():
    def __init__(self,n):
        self.p = list(range(n))
    def find(self,x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        if self.find(x) != self.find(y):
            self.p[self.find(y)] = self.find(x)
            return True
        else:
            return False
data1 = sys.stdin.read().strip().split()
idx = 0
while idx < len(data1):
    n = int(data1[idx])
    idx += 1
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(data1[idx]))
            idx += 1
        matrix.append(row)
    uf = UnionFind(n)
    data = []
    for _ in range(n):
        for j in range(_+1,n):
            data.append((matrix[_][j],_,j))
    data.sort()
    cost = 0
    count = 0
    for dis,x,y in data:
        if uf.union(x,y):
            cost += dis
            count += 1
            if count == n-1:
                break
    print(cost)
```
# DFS
思路还是很简单，就是一个点一个点搜。
## M18160:最大连通域面积
注意考虑面积为 $0$ 的情况。
```python
t = int(input())
def dfs(x,y):
    global pool,l
    pool[x][y][1] = 1
    l[c] += 1
    for a in range(-1,2):
        for b in range(-1,2):
            if (a != 0 or b != 0) and 0<=x+a<n and 0<=y+b<m and pool[x+a][y+b][0] == 1 and pool[x+a][y+b][1] == 0:
                dfs(x+a,y+b)
for _ in range(t):
    [n,m] = map(int,input().split())
    pool = []
    for i in range(n):
        data = input()
        row = []
        for j in range(m):
            if data[j] == ".":
                row.append([0,0])
            else:
                row.append([1,0])
        pool.append(row)
    c = -1
    l = []
    for i in range(n):
        for j in range(m):
            if pool[i][j][0] == 1 and pool[i][j][1] == 0:
                c += 1
                l.append(0)
                dfs(i,j)
    print(max(l) if l else 0)
```

与之类似的
## M05585:晶矿的个数
只是找的区域有两个种类，连通条件也不同
```python
t = int(input())
def dfs(x,y,k):
    mine[x][y][1] = 1
    for a,b in [(-1,0),(1,0),(0,-1),(0,1)]:
        if (a != 0 or b != 0) and 0<=x+a<n and 0<=y+b<n and mine[x+a][y+b][0] == k and mine[x+a][y+b][1] == 0:
            dfs(x+a,y+b,k)
for _ in range(t):
    n = int(input())
    mine = []
    for i in range(n):
        data = input()
        row = []
        for j in range(n):
            if data[j] == "r":
                row.append([1,0])
            elif data[j] == "b":
                row.append([2,0])
            else:
                row.append([0,0])
        mine.append(row)
    r,b = 0,0
    for i in range(n):
        for j in range(n):
            if mine[i][j][1] == 0:
                if mine[i][j][0] == 1:
                    r += 1
                    dfs(i,j,1)
                elif mine[i][j][0] == 2:
                    b += 1
                    dfs(i,j,2)
    print(r,b)
```
# BFS
学习了BFS，主要思想就是每次搜索一个点时，把他连接的下一级的几个点添加到搜索队列中。
## M19930:寻宝
没考虑到起点就是藏宝点，又被坑了。。。

```python
from collections import deque
[n,m] = map(int,input().split())
land = []
search = []
for i in range(n):
    land.append(list(map(int,input().split())))
    search.append([0]*m)
q = deque([(0,0,0)])
y = 0
while q:
    if land[0][0] == 1:
        y = 1
        print(0)
    i,j,d = q.popleft()
    search[i][j] = 1
    for a,b in [(-1,0),(1,0),(0,1),(0,-1)]:
        if 0<=i+a<n and 0<=j+b<m and search[i+a][j+b] == 0 and y == 0:
            if land[i+a][j+b] == 0:
                q.append((i+a,j+b,d+1))
            elif land[i+a][j+b] == 1:
                y = 1
                print(d+1)
if y == 0:
    print("NO")
```

# DP
## 02385:Apple Catching
设交换过 $w_{i}$ 次，在 $t_{i}$ 时间的最大苹果数为 $dp[t_{i}][w_{i}]$ ，可以写出 $dp[t_{i}][w_{i}]$ 关于 $dp[t_{i}-1][w_{i}]$ 和 $dp[t_{i}-1][w_{i}-1]$ 的转移方程。（太复杂，略去）
```python
t,w = map(int,input().split())
dp = [[(0,1)]*(w+2) for _ in range(t+1)]
for i in range(1,t+1):
    n = int(input())
    dp[i][1] = (dp[i-1][1][0]+int(n==1),1)
    for j in range(2,w+2):
        dp[i][j] = max((dp[i-1][j][0]+int(n==dp[i-1][j][1]),dp[i-1][j][1]),(dp[i-1][j-1][0]+int(n+dp[i-1][j-1][1]==3),3-dp[i-1][j-1][1]))
print(max(dp[-1])[0])
```

## M19929:环形穿梭车调度
本质就是一个加权的最长不减子序列问题。
```python
m,n = map(int,input().split())
list1 = list(map(int,input().split()))
w = list(map(int,input().split()))
ansl = [0]*m
ans = 0
for i in range(m):
    ansl[i] = w[i]
    for j in range(i):
        if list1[i] >= list1[j]:
            ansl[i] = max(ansl[i],ansl[j]+w[i])
    ans = max(ans,ansl[i])
print(ans)
```
