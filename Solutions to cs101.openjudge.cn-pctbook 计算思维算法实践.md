前言：你可能会遇到一些神秘的tag  
python function:说明python有自带的函数可以快速解决这个问题  
pypy:说明这个题用pypy可以显著减小运行时间。  
attention:说明这个题需要注意力。  
## E01003:Hangover 
http://cs101.openjudge.cn/pctbook/E01003/  
tags:implementation

预先打表处理，查询就比较方便（虽然理应用bisect）
```python
lengths = [0]#打表记录
for i in range(276):# 276是打表打出来的
    if i == 0:
        lengths.append(0.5)
    else:
        lengths.append(lengths[i] + 1 / (i+2))
while 1:
    a = input()
    if a == "0.00":
        break
    else:
        l = float(a)
        for i in range(len(lengths)-1):
            if lengths[i]<l<lengths[i+1]:
                print(f"{i+1} card(s)")
                break
```

## E01218:THE DRUNK JAILER
tags:math

我们知道在第 $m$ 轮时，只有 $m$ 的倍数的编号对应的牢房会改变门的开关状态。如果最后牢房的门是开的，说明门被开关了奇数次，也就说明这个数有奇数个不同的因数，只有完全平方数能做到这一点。
```python
for _ in range(int(input())):
    print(int(int(input())**0.5))
```

## E02676:整数的个数
http://cs101.openjudge.cn/pctbook/E02676/  
tags:python function

Counter练习题
```python
from collections import Counter
input()
dic = Counter(list(map(int,input().split())))
print(dic[1],dic[5],dic[10],sep = "\n")
```

## E02689:大小写字母互换
http://cs101.openjudge.cn/pctbook/E02689/  
tags: python function

swapcase(),lower(),upper()练习题
```python
print(input().swapcase())
```

## E02701:与7无关的数
http://cs101.openjudge.cn/pctbook/E02701/

tags:打表，implementation
```python
index = int(input())
print([1, 5, 14, 30, 55, 91, 91, 155, 236, 336, 457, 601, 770, 770, 995, 1251, 1251, 1575, 1936, 2336, 2336, 2820, 3349, 3925, 4550, 5226, 5226, 5226, 6067, 6967, 7928, 8952, 10041, 11197, 11197, 12493, 12493, 13937, 15458, 17058, 18739, 18739, 20588, 22524, 24549, 26665, 26665, 28969, 28969, 31469, 34070, 36774, 39583, 42499, 45524, 45524, 45524, 48888, 52369, 55969, 59690, 63534, 63534, 67630, 71855, 76211, 76211, 80835, 85596, 85596, 85596, 85596, 85596, 85596, 85596, 85596, 85596, 85596, 85596, 91996, 98557, 105281, 112170, 112170, 119395, 126791, 126791, 134535, 142456, 150556, 150556, 159020, 167669, 176505, 185530, 194746, 194746, 194746, 204547][index-1])
```

## E02712:细菌繁殖
http://cs101.openjudge.cn/pctbook/E02712/  
tags:implementation

此题重点在计算时间上
```python
n = int(input())
for i in range(n):
    data = list(map(int,input().split()))
    smonth,sday,num,emonth,eday = data
    days = 0
    months = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    if smonth == emonth:
        days = eday - sday
    else:
        days += months[smonth] - sday
        for j in range(smonth+1,emonth):
            days += months[j]
        days += eday
    print(num << days)
```

## E02724:生日相同
http://cs101.openjudge.cn/pctbook/E02724/  
tags:implementation

```python
dic = {}
for _ in range(int(input())):
    d = input().split()
    t = int(d[1])*100+int(d[2]) # 用100*月份+天数表征时间
    if t in dic:
        dic[t].append(d[0])
    else:
        dic[t] = [d[0]]
for m in range(1,13):
    for w in range(1,32):
        if 100*m+w in dic:
            if len(dic[100*m+w]) > 1:
                print(m,w,*dic[100*m+w])
```
## E02733:判断闰年
http://cs101.openjudge.cn/pctbook/E02733/  
tags:implementation

忽略题目中 $a<3000$ 的条件，使用4重if判断
```python
a = int(input())
if a % 4 == 0:
    if a % 100 == 0:
        if a % 400 == 0:
            if a % 3200 == 0:
                print("N")
            else:
                print("Y")
        else:
            print("N")
    else:
        print("Y")
else:
    print("N")
```

## E02734:十进制到八进制
http://cs101.openjudge.cn/pctbook/E02734/  
tags:implementation,python function

正常做法：
```python
a=int(input())
str1=""
while a>=8:
    str1+=str(a%8)
    a//=8
str1+=str(a)
print(int(str1[::-1]))
```

oct()函数练习题
```python
print(oct(int(input()))[2:])
```

## E23556:小青蛙跳荷叶
http://cs101.openjudge.cn/pctbook/E23556/
tags:dp
```python
jumps = int(input())
ways = [1, 2]
if jumps <= 2:
    print(ways[jumps-1])
else:
    for i in range(2, jumps):
        ways.append(ways[i-1] + ways[i-2])
    print(ways[jumps-1])
```

## M01002:方便记忆的电话号码
http://cs101.openjudge.cn/pctbook/M01002/
tags:implementation
```python
teledic = {}
nums = []
a = 0
for _ in range(int(input())):
    tnum = input().replace("-","").translate(str.maketrans("ABCDEFGHIJKLMNOPRSTUVWXY","222333444555666777888999"))
    nums.append(tnum)
    pnum = tnum[:3]+"-"+tnum[3:]
    if pnum in teledic:
        teledic[pnum]+=1
    else:
        teledic[pnum]=1
for i in sorted(nums):
    j = i[:3]+"-"+i[3:]
    if teledic[j] > 1:
        a = 1
        print(j,teledic[j])
        teledic[j] = 1
if a == 0:
    print("No duplicates.")
```

## M01017:装箱问题
http://cs101.openjudge.cn/pctbook/M01017/
tags:math,greedy

## M02746:约瑟夫问题
http://cs101.openjudge.cn/pctbook/M02746/
tags:math,implementation
如下的代码应该很直观了。
```python
while True:
    n,m = map(int, input().split())
    if n == 0 and m == 0:
        break
    else:
        monkeys = [i for i in range(n)]
        counts = 0
        for _ in range(n-1):
            counts += m-1
            counts = counts % len(monkeys)
            del monkeys[counts]
        print(monkeys[0] + 1)
```
**思考题**：
(1) 当 $0<m,n\leq 10^6$ 时，如何解决这个问题？
(2) 当 $0<m\leq 1000,0<n\leq 10^{12}$ 时，如何解决这个问题？

## M02810:完美立方
http://cs101.openjudge.cn/pctbook/M02810/
这是一个 $O(n^2)$ 算法。

```python
n = int(input())
dic = {}
for a in range(2,n+1):
    for b in range(a,n+1):
        if a**3+b**3 in dic:
            dic[a**3+b**3].append([a,b])
        else:
            dic[a**3+b**3] = [[a,b]]
for d in range(2,n+1):
    aa = []
    for c in range(d-1,1,-1):
        if d**3-c**3 in dic:
            for di in dic[d**3-c**3]:
                if c >= di[1]:
                    aa.append([di[0],di[1],c,d])
    aa.sort(key = lambda x:x[0])
    for k in aa:
        print(f"Cube = {k[3]}, Triple = ({k[0]},{k[1]},{k[2]})")
```


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

## T01011:Sticks
http://cs101.openjudge.cn/pctbook/T01011/  
tags:dfs,剪枝

超级剪枝题目
```python
import sys
sys.setrecursionlimit(10**7)
input = sys.stdin.readline
def judge(arr,k):
    n = len(arr)
    s = sum(arr)
    target = s // k
    ALL_USED = (1 << n) - 1
    fail = set()
    def dfs(used_mask,cur,si,buckets):
        if buckets == 1:
            return True
        if used_mask in fail:
            return False
        if cur == target:
            res = dfs(used_mask,0,0,buckets-1)
            if not res:
                fail.add(used_mask)
            return res
        last = -1
        i = si
        while i < n:
            if (used_mask >> i) & 1 == 0:
                v = arr[i]
                if v != last and cur + v <= target:
                    next_mask = used_mask | (1 << i)
                    if dfs(next_mask,cur+v,i+1,buckets):
                        return True
                    last = v
                    if cur == 0:
                        fail.add(used_mask)
                        return False
            i += 1
        fail.add(used_mask)
        return False
    return dfs(0,0,0,k)
def solve():
    while True:
        n = int(input())
        if n == 0:
            break
        arr = list(map(int, input().split()))
        arr.sort(reverse=True)
        s = sum(arr)
        found = False
        for k in range(min(n, s // arr[0]),1,-1):
            if s % k != 0:
                continue
            if judge(arr,k):
                print(s//k)
                found = 1
                break
        if not found:
            print(s)
if __name__ == "__main__":
    solve()
```

## T25353:排队
http://cs101.openjudge.cn/pctbook/T25353/
```python
import sys
input = sys.stdin.readline
n,d = map(int,input().split())
nums = [int(input()) for _ in range(n)]
i = 0
left = nums.copy()
while left:
    ll = len(left)
    maxn,minn = left[0],left[0]
    cand = []
    idx = -1
    for j in range(ll):  
        if j == 0 or maxn - d <= left[j] <= d + minn:
            cand.append(j)
        if left[j] < minn:
            minn = left[j]
        if left[j] > maxn:
            maxn = left[j]
    print("\n".join(map(str,sorted([left[i] for i in cand]))),end = "\n")
    sc = set(cand)
    left = [left[i] for i in range(ll) if i not in sc]
```


## T25561:2022决战双十一
http://cs101.openjudge.cn/pctbook/T25561/  
tags:brute force,dfs

dfs题目
```python
import sys
sys.setrecursionlimit(1<<30)
n,m = map(int,input().split())
prices = []
discounts = []
for _ in range(n):
    dic = {}
    data = input().split()
    for d in data:
        d1,d2 = map(int,d.split(":"))
        dic[d1] = d2
    prices.append(dic)
for _ in range(m):
    dic = {}
    data = input().split()
    for d in data:
        d1,d2 = map(int,d.split("-"))
        dic[d1] = d2
    discounts.append(dic)
def dfs(i,pay = [0]*m):
    if i == n:
        s = sum(pay)
        total = s - s//300*50
        dsum = 0
        for j in range(m):
            mdis = 0
            for disc in discounts[j]:
                if pay[j] >= disc:
                    mdis = max(mdis,discounts[j][disc])
            dsum += mdis
        return total - dsum
    ans = 10**9
    for k in prices[i]:
        pay[k-1] += prices[i][k]
        ans = min(ans,dfs(i+1,pay))
        pay[k-1] -= prices[i][k]
    return ans
print(dfs(0))
```

## T25573:红蓝玫瑰
http://cs101.openjudge.cn/pctbook/T25573/  
tags:dp,greedy

核心思想：从右到左遍历，如果遇到一朵蓝色花就直接改变颜色；如果有多朵连续的蓝色花就翻转一次颜色把他们都变成红色。
```python
r = input()
dic={"R":0,"B":1} # 将红色和蓝色转换为0和1
ans = 0
flip = 0 # 翻转次数
i = len(r)-1 # 从右到左开始遍历
while i >= 0:
    if flip % 2 == dic[r[i]]: # 经过了flip次翻转变成了红色
        while i >= 0 and flip % 2 == dic[r[i]]:
            i -= 1 # 向左遍历直到出现蓝色
    else:
        l = 0 # 统计蓝色长度
        while i >= 0 and flip % 2 != dic[r[i]]:
            i -= 1
            l += 1
        if l != 1: # 只有一朵蓝色玫瑰时，直接改变颜色最优，其他情况下翻转一次最优
            flip += 1
        ans += 1 # 操作次数加一
print(ans)
```

## T26971:分发糖果
http://cs101.openjudge.cn/pctbook/T26971/  
tags:greedy
```python
n = int(input())
rat = list(map(int,input().split()))
cand = [0]*n
q = []
for i in range(n):
    if (i == 0 or rat[i]<=rat[i-1]) and (i == n-1 or rat[i]<=rat[i+1]):
        cand[i] = 1
        q.append(i)
for i in range(n-1):
    if rat[i+1] > rat[i]:
        cand[i+1] = max(cand[i]+1,cand[i+1])
    elif rat[i+1] == rat[i]:
        cand[i+1] = max(1,cand[i+1])
for i in range(n-1,0,-1):
    if rat[i-1] > rat[i]:
        cand[i-1] = max(cand[i]+1,cand[i-1])
    elif rat[i-1] == rat[i]:
        cand[i-1] = max(1,cand[i-1])
print(sum(cand))
```

## T27103:最短的愉悦旋律长度
http://cs101.openjudge.cn/pctbook/T27103/  
tags:math,dp

如果这个题你的代码长度超过了300B，那就比较失败了。  
思路：如果长度为 $n-1$ 的所有旋律已经在前面的数全部出现过，那么后面的数能够让长度为 $n$ 的所有旋律均出现的充要条件就是后面的数必须出现所有不同的音符。所以我们只要用一个set记录所有出现过的音符即可。
```python
a=lambda:map(int,input().split())
n,m=a()
d=list(a())
c=set()
ans=1
for i in d:
    c.add(i)
    if len(c)==m: # 已经出现过所有音符，愉悦旋律长度+1 
        c=set()
        ans+=1
print(ans)
```

## T27104:世界杯只因
http://cs101.openjudge.cn/pctbook/T27104/
tags:greedy
依旧是区间覆盖题目

## T29947:校门外的树又来了
http://cs101.openjudge.cn/pctbook/T29947/  
tags:interval merging

排序是这个方法的点睛之笔。
```python
l,m=map(int,input().split())
data = [list(map(int,input().split())) for _ in range(m)]
data.sort() # 按照起点排序
[s,e] = data[0] # s,e分别表示当前区间的起点和终点
a = s # 处理最左端的树
for d in data:
    if d[0] > e: # 如果要处理的区间起点在当前区间终点右边，说明当前区间无法被合并了
        s = d[0]
        a += (s-e-1) # 增加不在区间内的长度（树的数量）
        e = d[1]
    else:
        e = max(e,d[1]) # 更新当前区间终点
a += l-e # 处理最右端的树
print(a)
```
