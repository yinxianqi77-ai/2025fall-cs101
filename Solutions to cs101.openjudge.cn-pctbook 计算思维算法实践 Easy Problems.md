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
tags:implementation,python function

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

calendar库练习题
```python
import calendar as c
print("Y" if c.isleap(int(input())) else "N")
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

## E02750:鸡兔同笼
http://cs101.openjudge.cn/pctbook/E02750/  
tags:math

如果给定的数是奇数，则不可能；如果是偶数，输出 (ceil(a//4),a//2) 。
```python
a=int(input())
if a%2==0:
    print(-((-a)//4),a//2)
else:
    print(0,0)
```

## E02753:菲波那契数列
http://cs101.openjudge.cn/pctbook/E02753/  
tags:dp，打表  

最短代码，不服来战。
```python
arr = [1,1]
for i in range(20):
	arr.append(arr[-1]+arr[-2]) # 递推式a_n= a_{n-1}+a_{n-2}
for _ in range(int(input())):
	print(arr[int(input())-1])
```

## E02767:简单密码
http://cs101.openjudge.cn/pctbook/E02767/  
tags:implementation,python function

translate()练习题
```python
cypher = input()
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
rolled_alphabet = alphabet[-5:] + alphabet[:-5]
decripted = cypher.translate(
    str.maketrans(alphabet, rolled_alphabet)
)
print(decripted)
```

## E02792:集合加法
http://cs101.openjudge.cn/pctbook/E02792/  
tags:brute force,dict,python function,pypy

利用字典里查找为 O(1) 的特性，极大压缩时间。
```python
from collections import Counter
for _ in range(int(input())):
    s = int(input())
    a = int(input())
    alist = list(map(int,input().split()))
    b = int(input())
    blist = list(map(int,input().split()))
    dic = Counter(alist) # 利用Counter构建字典
    num = 0
    for m in blist:
        if s-m in dic:
            num += dic[s-m]
    print(num)
```

小技巧：  
 1. python自带的函数永远比手动实现快。
 2. 在多重循环且执行的语句**具有明显重复性**时，pypy会非常快。
```python
# python 5390ms AC
# pypy 8981ms TLE
for _ in range(int(input())):
    s = int(input())
    input()
    alist = list(map(int,input().split()))
	input()
    blist = list(map(int,input().split()))
    num = 0
    for m in blist:
        num += alist.count(s-m) # 使用列表的count函数，不具有重复性
        # 并且count函数是使用C实现的，所以pypy并不会有时间上的帮助
    print(num)
```

```python
# python 29940ms TLE
# pypy 1416ms AC
for _ in range(int(input())):
    s = int(input())
    a = int(input())
    alist = list(map(int,input().split()))
    b = int(input())
    blist = list(map(int,input().split()))
    num = 0
    for m in blist:
        for n in alist:
            num += m+n==s # 手动实现，具有重复性
    print(num)
```

## E02804:词典
http://cs101.openjudge.cn/pctbook/E02804/  
tags:implementation,data structure

```python
dic = {}
while True:
    stri = input()
    if stri == "":
        break
    else:
        string = stri.split()
        if string[1] in dic:
            continue
        else:
            dic[string[1]] = string[0]
try:
    while True:
        need = input()
        if need in dic:
            print(dic[need])
        else:
            print("eh")
except EOFError:
    pass
```

## E02808:校门外的树
http://cs101.openjudge.cn/pctbook/E02808/  
tags:implementation

用一个列表模拟砍树过程。
```python
l,n = map(int,input().split())
trees = [1]*(l+1)
for i in range(n):
    s,e = map(int,input().split())
    for j in range(s,e+1):
        trees[j] = 0
print(sum(trees))
```

## E02883:Checking order
http://cs101.openjudge.cn/pctbook/E02883/  
tags:python function

sorted()函数练习题
```python
import sys
for nums in sys.stdin.read().splitlines():
    numed = list(map(int,nums.split()))
    if sorted(numed) == numed:
        print("Yes")
    else:
        print("No",*sorted(numed))
```

## E02899:矩阵交换行
http://cs101.openjudge.cn/pctbook/E02899/  
tags:implementation

格式练习题
```python
matrix = []
for _ in range(5):
    matrix.append(list(map(int,input().split())))
n, m = map(int, input().split())
if n < 0 or n >= 5 or m < 0 or m >= 5:
    print("error")
else:
    matrix[n],matrix[m] = matrix[m],matrix[n] # 交换元素
    for i in matrix:
        print("".join([f"{num:4d}" for num in i]))
```

## E02910:提取数字
http://cs101.openjudge.cn/pctbook/E02910/   
tags:implementation,python function

isdigit()练习题。
```python
string = input().strip() + "x"
arr = []
a = ""
for s in string:
    if s.isdigit():
        a += s
    else:
        if a:
            arr.append(int(a))
        a = ""
print(*arr,sep="\n")
```

## E02911:受限完全平方数
http://cs101.openjudge.cn/pctbook/E02911/   
tags:打表

请输入文本。
```python
max = int(input())
for i in [3136,4489]:
    if i < max :
        print(i)
```

## E02913:加密技术
http://cs101.openjudge.cn/pctbook/E02913/   
tags:implementation

这个题就不好用translate()了，直接来。
```python
string = input()
encoded = ""
num = "4962873"
for i in range(len(string)):
    encoded += chr(ord(string[i]) + int(num[i % 7]))
print(encoded)
print(string)
```

## E02927:判断数字个数
http://cs101.openjudge.cn/pctbook/E02927/   
tags:implementation,python function

isdigit()练习题。
```python
import sys
strings = sys.stdin.read().splitlines()
for s in strings:
    times = [0] * 10
    for n in s:
        if n.isdigit():
            times[int(n)] += 1
    for i in range(10):
        if times[i] != 0:
            print(f"{i}:{times[i]}")
```

依旧Counter()练习题，注意key是字符串。
```python
import sys
from collections import Counter
for s in sys.stdin.read().splitlines():
    c = Counter(s)
    for i in range(10):
        if str(i) in c:
            print(f"{i}:{c[str(i)]}")
```

## E02936:试剂配制
http://cs101.openjudge.cn/pctbook/E02936/   
tags:implementation

```python
n = int(input())
nums = list(map(int,input().split()))
if 1 in nums and 2 in nums:
    print(0)
elif 3 in nums and 4 in nums:
    print(0)
elif 5 in nums and 6 not in nums:
    print(0)
elif 6 in nums and 5 not in nums:
    print(0)
elif 7 not in nums and 8 not in nums:
    print(0)
else:
    print(1)
```

## E02940:求和
http://cs101.openjudge.cn/pctbook/E02940/   
tags:implementation

```python
a=input().split()
num = 0
for n in range(1,int(a[1])+1):
    num += int(a[0]) * int(n * "1")
print(num)
```

## E02942:吃糖果
http://cs101.openjudge.cn/pctbook/E02942/   
tags:recursion

```python
def f(num):
    if num == 1:
        return 1
    elif num == 2:
        return 2
    return f(num-1)+f(num-2)
print(f(int(input())))
```

## E02946:玩游戏
http://cs101.openjudge.cn/pctbook/E02946/   
tags:implementation

```python
k,N = map(int,input().split())
current = k
for _ in range(N):
    op, num_str = input().split()
    a = int(num_str)
    if op == 'plus':
        current += a
    elif op == 'minus':
        current -= a
    elif op == 'multiply':
        current *= a
print(current)
```

## E03143:验证“歌德巴赫猜想”
http://cs101.openjudge.cn/pctbook/E03143/   
tags:math,brute force

先生成给定范围内的质数，再枚举判断即可。
```python
def sieve(n):# 筛法生成素数
    mask = [1] * (n + 1)
    mask[0] = mask[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if mask[i]:
            mask[i*i::i] = [0] * len(mask[i*i::i])
    return [i for i in range(n + 1) if mask[i]]
num = int(input())
primes = sieve(num)
pset = set(primes)
if num < 6 or num % 2 != 0:
    print("Error!")
else:
    for i in primes:
        if i <= num //2 and num - i in pset:
            print(f"{num}={i}+{num-i}")
```

## E03225:满足条件的整数
http://cs101.openjudge.cn/pctbook/E03225/   
tags:brute force

在 $i,j$ 较小时可以直接用 $\sqrt{ i^{2}+j^{2} }$ 是否是整数来直接判断。
```python
from math import sqrt
for i in range(1,101):
    for j in range(i, 101):
    a = sqrt(i**2 + j**2)
        if a == int(a) and a<=100:
            print(str(i) + "*" + str(i)+" + " + str(j) + "*" + str(j) + " = " + str(int(a)) + "*"+ str(int(a)))
```
## E03248:最大公约数
http://cs101.openjudge.cn/pctbook/E03248/  
tags:math,python function

可以用辗转相除法，也可以直接用gcd()。
```python
import sys,math
for s in sys.stdin.readlines():
    a,b = map(int,s.split())
    print(math.gcd(a,b))
```

## E03406:书架
http://cs101.openjudge.cn/pctbook/E03406/   
tags:sortings

排序再累加判断即可。
```python
n,b = list(map(int,input().split()))
cows = [int(input()) for _ in range(n)]
cows = sorted(cows)[::-1]
sum = 0
nums = 0
for i in range(n):
    sum += cows[i]
    nums += 1
    if sum >= b:
        print(nums)
        break
```

## E03670:计算鞍点
http://cs101.openjudge.cn/pctbook/E03670/   
tags:matrix

直接做就行了。
```python
nums = [list(map(int,input().split())) for _ in range(5)]
is_found = 0
for i in range(5):
    n = max(nums[i])
    index = nums[i].index(n)
    if max(nums[i]) == min(nums[0][index],nums[1][index],nums[2][index],nums[3][index],nums[4][index]):
        is_found = 1
        print(f"{i+1} {index+1} {n}")
if is_found == 0:
    print("not found")
```
## E04018:子串
http://cs101.openjudge.cn/pctbook/E04018/   
tags:two pointers

很经典的双指针题目
```python
import sys
for si in sys.stdin.read().splitlines():
    t,s = si.split()
    lt,ls = len(t),len(s)
    j = 0
    found = 1
    for i in range(lt):
        while j < ls and s[j] != t[i]:
            j += 1
        if j == ls:
            found = 0
            break
        j += 1
    print("Yes" if found else "No")
```

进阶练习：
https://codeforces.com/problemset/problem/2174/A

## E04067:回文数字（Palindrome Number）
http://cs101.openjudge.cn/pctbook/E04067/   

切片常见技巧。
```python
import sys
data = sys.stdin.read().splitlines()
for line in data:
    if line[::-1] == line:
        print("YES")
    else:
        print("NO")
```

## E04138:质数的和与积
http://cs101.openjudge.cn/pctbook/E04138/

用筛法得到质数后直接实现就行了。
```python
# StarrySky
def sieve1(n):
    mask = [1] * (2*n + 1)
    counter = 2
    mask[0] = 0
    mask[1] = 0
    prime = [counter]
    while counter < n:
        mask[counter * 2::counter] = [0] * (len(mask[counter * 2::counter]))
        counter += 1
        while mask[counter] == 0:
            counter += 1
        prime.append(counter)
    return prime
num = int(input())
list1 = sieve1(num*2)
if num % 2 == 1:
    print(2*(num-2))
else:
    for i in range(num//2):
        if num//2-i in list1 and num//2+i in list1:
            print((num//2-i)*(num//2+i))
            break
```

## E04146:数字方格
http://cs101.openjudge.cn/pctbook/E04146/   
tags:number theory,implementation

这个题当然可以直接做。
```python
# DeepSeek 168ms
n = int(input().strip())
max_sum = -1
# 遍历所有可能的a1, a2, a3值
for a1 in range(n + 1):
    for a2 in range(n + 1):
        for a3 in range(n + 1):
            # 检查三个条件
            if (a1 + a2) % 2 == 0 and (a2 + a3) % 3 == 0 and (a1 + a2 + a3) % 5 == 0:
                current_sum = a1 + a2 + a3
                if current_sum > max_sum:
                    max_sum = current_sum
# 输出结果
print(max_sum)
```

不过这个题其实有 $O(1)$ 的方法，观察发现 $n=15$ 时，我们取 $a_{1}=a_{2}=a_{3}=15$ ，可以满足三个条件，容易证明这是可能最大值。利用余数的可加性，我们可以设 $n=15k+r$，则此时的答案 $f(n)$ 就等于 $45k+f(r)$ （可以自行代回去检验），所以只需要求解 $n<15$ 的情形就可以了。
```python
pre_list = [0,0,5,5,10,10,15,15,20,25,25,30,30,35,40]
a = int(input())
print(pre_list[a % 15] + 45 * (a // 15))
```

## E06374:文字排版
http://cs101.openjudge.cn/pctbook/E06374/   
tags:implementation,string

注意边界。
```python
n = int(input())
text = input().split()
num = 0
for i in range(n):
    num += len(text[i])+1
    if i == n-1:
        print(text[i])
    elif num + len(text[i+1]) > 80:
        print(text[i],end = "\n")
        num =0
    else:
        print(text[i],end = " ")
```

## E07618:病人排队
http://cs101.openjudge.cn/pctbook/E07618/  
tags:sortings

这个题先将老年人和青年人分开，然后按照老年人年龄排序。（注意sort()不会改变输入先后顺序！）
```python
n = int(input())
old = []
young = []
for _ in range(n):
    data = input().split()
    if int(data[1])>=60:
        old.append([data[0],int(data[1])])
    else:
        young.append(data[0])
old = sorted(old,key = lambda x:x[1],reverse=True)
for o in old:
    print(o[0])
for y in young:
    print(y)
```

## E07743:计算矩阵边缘元素之和
http://cs101.openjudge.cn/pctbook/E07743/   
tags:implementation,matrix

可以直接实现。
```python
m,n = map(int,input().split())
arr = [list(map(int,input().split())) for _ in range(m)]
ans = 0
for i in range(m):
    for j in range(n):
        if i == 0 or i == m-1 or j == 0 or j == n-1:
            ans += arr[i][j]
print(ans)
```

也可以直接加上下左右四条边。注意m或n等于1的情况。
```python
m,n = map(int,input().split())
matrix = [list(map(int,input().split())) for _ in range(m)]
if m == 1:
    sums = sum(matrix[0])
else:
    sums = sum(matrix[0])+sum(matrix[m-1])
for i in range(1,m-1):
    if n == 1:
        sums+=matrix[i][0]
    else:
        sums += matrix[i][0]+matrix[i][n-1]
print(sums)
```

## E07810:19岁生日礼物-Birthday Gift
http://cs101.openjudge.cn/pctbook/E07810/

```python
# 神秘一行流
print("\n".join(["Yes" if int(string) % 19 == 0 or "19" in string else "No" for string in [input() for _ in range(int(input()))]]))
```

## E08219:判断数正负
http://cs101.openjudge.cn/pctbook/E08219/

哈哈。
```python
N = int(input())
if N > 0:
    print("positive")
elif N == 0:
    print("zero")
else:
    print("negative")
```

## E12556:编码字符串
http://cs101.openjudge.cn/pctbook/E12556/   
tags:implementation,python function,string

直接实现即可。
```python
str1 = input().lower()
ref=str1[0]
out="(" + ref + ","
n=0
for i in range(len(str1)):
    if str1[i] != ref :
        out+= str(i-n) + ")"+"(" + str1[i] +"," 
        ref=str1[i]
        n=i
out+= str(len(str1)-n) + ")"
print(out)
```

## E18161:矩阵运算(先乘再加)
http://cs101.openjudge.cn/pctbook/E18161/   
tags:matrix

```python
size_info = []
def get_matrix():# 输入矩阵
    [n,m] = map(int,input().split())
    size_info.append([n,m])
    matrix = []
    for _ in range(n):
        matrix.append(list(map(int,input().split())))
    return matrix
A = get_matrix()
B = get_matrix()
C = get_matrix()
# 判断运算是否合法
if size_info[0][1] != size_info[1][0] or size_info[0][0] != size_info[2][0] or size_info[1][1] != size_info[2][1]:
    print("Error!")
else:
    for i in range(size_info[2][0]):
        for j in range(size_info[2][1]):
            n = 0
            for k in range(size_info[0][1]):
                n += A[i][k] * B[k][j]
            C[i][j] += n
    for i in range(size_info[2][0]):
        print(*C[i])
```

## E18188:图像的均值滤波
http://cs101.openjudge.cn/pctbook/E18188/  
tags:matrix

注意判断周围8个方块时的写法。
```python
n,m = map(int,input().split())
cells = []
cells1 = []
for i in range(n):
	data = list(map(int,input().split()))
    cells.append(data)
    cells1.append(data)
for i in range(n):
    for j in range(m):
        sum = 0
        blocks = 0
        for k in [-1,0,1]:
            for l in [-1,0,1]:
                if i+k >= 0 and i+k < n and j+l >=0 and j+l < m :
                    sum += cells1[i+k][j+l]
                    blocks += 1
        cells[i][j] = sum // blocks
for i in range(n):
    print(*cells[i])
```

## E18223:24点
http://cs101.openjudge.cn/pctbook/E18223/  
tags:python function

利用product函数快速解决。
```python
from itertools import product
for _ in range(int(input())):
    a,b,c,d = map(int,input().split())
    yes = 0
    for x,y,z,w in product((-1,1),(-1,1),(-1,1),(-1,1)):
        if x*a+y*b+z*c+w*d == 24:# 因为是4个正数，不用担心4个负号的情况
            yes = 1
            break
    print("YES" if yes else "NO")
```

## E18224:找魔数
http://cs101.openjudge.cn/pctbook/E18224/  
tags:python function

二进制，八进制，十六进制有对应函数。

```python
set1 = set()# 用set装所有魔数
for i in range(1,32):
    for j in range(1,int((1000-i**2)**0.5)+1):
        set1.add(i**2+j**2)
n = int(input())
nums = list(map(int,input().split()))
for i in nums:
    if i in set1:
        print(bin(i).lower(),oct(i).lower(),hex(i).lower())
```

## E19942:二维矩阵上的卷积运算
http://cs101.openjudge.cn/pctbook/E19942/   
tags:matrix

直接算。
```python
m,n,p,q = map(int,input().split())
matrix = []
core = []
sum = [[0] * (n+1-q) for _ in range(m+1-p)]
for i in range(m):
    matrix.append(list(map(int,input().split())))
for i in range(p):
    core.append(list(map(int,input().split())))
for i in range(m+1-p):
    for j in range(n+1-q):
        for k in range(p):
            for l in range(q):
                sum[i][j] += core[k][l] * matrix[i+k][j+l]
for i in range(m+1-p):
    print(*sum[i])
```

## E19943:图的拉普拉斯矩阵
http://cs101.openjudge.cn/pctbook/E19943/  
tags:matrix

```python
n,m = map(int,input().split())
l_matrix = [[0] * n for i in range(n)]
for i in range(m):
    x1,x2 = list(map(int,input().split()))
    l_matrix[x1][x1] += 1
    l_matrix[x2][x2] += 1
    l_matrix[x1][x2] = -1
    l_matrix[x2][x1] = -1
for i in range(n):
    print(*l_matrix[i])
```

## E19949:提取实体
http://cs101.openjudge.cn/pctbook/E19949/  
tags:implementation,string

注意几个连着的###算是一个实体。
```python
n = int(input())
entities = 0
for _ in range(n):
    words = input().split()
    can_add = 1
    for word in words:
        if word[:3] == "###" and word[-3:] == "###":
            if can_add:
                entities += 1
                can_add = 0
        else:
            can_add = 1
print(entities)
```

## E20742:泰波拿契數
http://cs101.openjudge.cn/pctbook/E20742/   
tags:dp

递推公式都给你了那还说啥了。
```python
t = [0,1,1]
for i in range(int(input())-2):
    t.append(t[-1]+t[-2]+t[-3])
print(t[-1])
```

## E21459:How old are you？
http://cs101.openjudge.cn/pctbook/E21459/  
tags:implementation

善用f-string和format()
```python
num = int(input())
while num !=1:
    if num%2==1:
        print("{}*3+1={}".format(num, num*3+1))
        num = num*3+1
    else:
        print("{}/2={}".format(num, num//2))
        num = num//2
```

## E21727:湾仔码头
http://cs101.openjudge.cn/pctbook/E21727/   
tags:greedy

显然的greedy题目。
```python
n,m = map(int,input().split())
bricks = input().split()
v = 0
num = 0
for b in bricks:
    v += int(b)
    if v <= m:
        num += 1
    else:
        break
print(num)
```

## E21728:排队做实验
http://cs101.openjudge.cn/pctbook/E21728/   
tags:implementation

```python
n = int(input())
a = list(map(int,input().split()))
b = list(map(int,input().split()))
print(f"{sum([a[b[i]-1]*(n-i-1) for i in range(n)])/n:.2f}")
```

## E22271:绿水青山之植树造林活动
http://cs101.openjudge.cn/pctbook/E22271/   
tags:implementation

使用字典可轻松解决。
```python
n = int(input())
fruits = {}
for i in range (n):
    fruit = input()
    if fruit in fruits:
        fruits[fruit] += 1
    else:
        fruits[fruit] = 1
fruits_s = sorted(fruits.keys())
for fruit in fruits_s:
    print(f"{fruit} {100*fruits[fruit]/n:.4f}" + "%")
```

## E22548:机智的股民老张
http://cs101.openjudge.cn/pctbook/E22548/   
tags:greedy,divide and conquer

最开始，想了一个 $O(n\log n)$ 的方法，使用分治。核心思想就是将数组分为左右两半，根据左右两半数组的利润最大值得到整体数组的利润最大值。
```python
num_list = list(map(int,input().split()))
def find_max_diff(nums):
    nums1 = nums[:len(nums) // 2]
    nums2 = nums[len(nums) // 2:len(nums)]
    if len(nums1) == 1:
        return 0
    elif len(nums1) == 2:
        if nums1[1]>nums1[0]:
            return nums1[1]-nums1[0]
        else:
            return 0
    elif len(nums2) == 2:
        if nums2[1]>nums2[0]:
            return nums2[1]-nums2[0]
        else:
            return 0
    return max(max(nums2)-min(nums1),find_max_diff(nums1),find_max_diff(nums2))
if sorted(num_list) == num_list[::-1]:
    print(0)
else:
    print(find_max_diff(num_list))
```

后来发现，用贪心可以给出 $O(n)$ 的方法。将利润最大化，无非就是干两件事：使买入价尽量低和使卖出价尽量高。于是我们可以这么做：  
从左到右遍历数组，同时同步更新当前买入价的最小值mins和当前利润最大值ans。每新增进来一个值a，把mins更新为min(mins,a)（这没啥说的），然后把当前能获得的最大利润ans更新为max(ans,a-mins)（因为a-mins是以a为卖出价格时的最大利润）。最后得到的ans就是答案。
代码实现：
```python
arr = list(map(int,input().split()))
mins,ans = 10**4,0
for a in arr:
    mins = min(mins,a)
    ans = max(ans,a-mins)
print(ans)
```

## E23555:节省存储的矩阵乘法
http://cs101.openjudge.cn/pctbook/E23555/   
tags:implementation,matrix


将三元组形式转化为矩阵，相乘之后再转化回三元组形式。
```python
n,m1,m2 = map(int,input().split())
a = [[0]*n for _ in range(n)]
b = [[0]*n for _ in range(n)]
for _ in range(m1):
    [i,j,v] = map(int,input().split())
    a[i][j] = v
for _ in range(m2):
    [i,j,v] = map(int,input().split())
    b[i][j] = v
for i in range(n):
    for j in range(n):
        sums = 0
        for k in range(n):
            sums += a[i][k]*b[k][j]
        if sums != 0:
            print(f"{i} {j} {sums}")
```

## E23556:小青蛙跳荷叶
http://cs101.openjudge.cn/pctbook/E23556/  
tags:dp

简单dp问题。青蛙在第n片荷叶时它的上一步只能从第n-1片荷叶或者第n-2片荷叶出发，所以可以写出递推式，进而求解。
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

## E23563:多项式时间复杂度
http://cs101.openjudge.cn/pctbook/E23563/   
tags:implementation,string

先用“+”号分离多项式的每一项再找最高次数。
```python
text = input().split("+")
max_pot = 0
for i in range(len(text)):
    if text[i][0] != "0":
        max_pot = max(max_pot,int(text[i].split("^")[1]))
if max_pot == 0:
    print(1)
else:
    print(f"n^{max_pot}")
```

## E23564:数论
http://cs101.openjudge.cn/pctbook/E23564/   
tags:number theory

```python
def sieve(n):# 用筛法生成质数
    mask = [1] * (n+1)
    mask[0] = 0
    mask[1] = 0
    for i in range(2,int(n**0.5)+1):
        if mask[i]:
            mask[i*i:n+1:i] = [0] * len(mask[i*i:n+1:i])
    return [i for i in range(2, n + 1) if mask[i]]
n = int(input())
prime = sieve(n)
prime_list = []
a = 1
if n == 1 :
    print(1)
else:
    for p in prime:
        if n % (p**2) == 0:# 如果有平方质因数
            a = 0
            print(0)
            break
        elif n % p == 0:
            prime_list.append(p)
    if a :
        if len(prime_list) % 2 == 0:# 有偶数个质因数
            print(1)
        else:# 有奇数个质因数
            print(-1)
```

## E23566:决战双十一
http://cs101.openjudge.cn/pctbook/E23566/   
tags:implementation

```python
n,m = map(int,input().split())
data = [[] for _ in range(m)]# 记录打折前第i家店铺的总花费
sums = 0# 打折前总花费
dis2 = 0# 每家店铺的打折总额
for _ in range(n):
    price = list(map(int,input().split()))
    sums += price[1]
    if data[price[0]-1]:
        data[price[0] - 1][0] += price[1]
    else:
        data[price[0] - 1].append(price[1])
for i in range(m):
    discount = list(map(int,input().split("-")))
    if data[i][0] >= discount[0]:# 如果满足第i家店铺的打着要求
        dis2 += discount[1]
dis1 = sums // 200 * 30# 总体的打折
print(sums - dis1 - dis2)
```

## E24684:直播计票
http://cs101.openjudge.cn/pctbook/E24684/  
tags:data structure

依旧Counter()练习题。
```python
from collections import Counter
data = Counter(map(int,input().split()))# 记录所有选项的出现次数
tgt = max(data.values())# 最多的出现次数
ans = []
for d in data:
    if data[d] == tgt:
        ans.append(d)
print(*sorted(ans))
```

## E25538:二进制回文的整数
http://cs101.openjudge.cn/pctbook/E25538/   
tags:implementation

```python
n = int(input())
string = bin(n)[2:]
if string == string[::-1]:
    print("Yes")
else:
    print("No")
```

## E25580:木板掉落
http://cs101.openjudge.cn/pctbook/E25580/   
tags:implementation

假设至少需要挡住的最小速度为 $v$ ，那么当木板恰好能挡住时，有
$$
0.5gt^2=H-h
$$
和
$$
vt=L
$$
联立解得    
$$
h=H-0.5g\left( \frac{L}{v} \right)^{2}
$$
```python
a = lambda:map(int,input().split())
h,l,n = a()
v = sorted(a())
print(f"{h-5*(l/v[n//2])**2:.2f}")
```

## E27273:简单的数学题
http://cs101.openjudge.cn/pctbook/E27273/   
tags:implementation,math

```python
powers_of_two = [2**i for i in range(20)]# 预先计算2的幂次
for i in range(int(input())):
    num = int(input())
    ans = num*(num + 1)//2#高斯求和公式
    # 减去2的幂次
    for element in powers_of_two:
        if element <= num:
            ans -= 2 * element
    print(ans)
```

更加数学一点的方式：利用
$$
\sum_{i=0}^n 2^i=2^{n+1}-1
$$
```python
from math import log2
for _ in range(int(input())):
    n = int(input())
    print((n*(n+1))//2-2*((2**(int(log2(n))+1)-1)))
    # 2**(int(log2(n))是小于n的2的整数次幂中最大的。
```

## E27301:给植物浇水
http://cs101.openjudge.cn/pctbook/E27301/      
tags:implementation,two pointers

注意处理两者相遇时的细节。
```python
n,alice,bob = map(int,input().split())
trees = list(map(int,input().split()))
refill,left,right = 0,0,n-1
while left <= right:
    if left < right:# 两者没有相遇
	    # Alice浇水
        if alice < trees[left]:
            alice = a
            refill += 1
        alice -= trees[left]
        left += 1
	    # Bob浇水
        if bob < trees[right]:
            bob = b
            refill += 1
        bob -= trees[right]
        right -= 1
    else:
        if alice >= bob:
           if alice < trees[left]:
                refill += 1
        elif bob < trees[right]:
                refill += 1
        left += 1
        right -= 1
print(refill)
```

## E27653:Fraction类
http://cs101.openjudge.cn/pctbook/E27653/   
tags:implementation

```python
from math import gcd
class Fraction:# Fraction类
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def show(self):
        return str(self.a)+"/"+str(self.b)
def add_fractions(f1,f2):# 实现分数加法
    return Fraction((f1.a*f2.b+f1.b*f2.a)//gcd(f1.b,f2.b),f1.b*f2.b//gcd(f1.b,f2.b))
fracs = input().split()
f1=Fraction(int(fracs[0]),int(fracs[1]))
f2=Fraction(int(fracs[2]),int(fracs[3]))
print(add_fractions(f1,f2).show())
```

## E28336:消消乐
http://cs101.openjudge.cn/pctbook/E28336/   
tags:data structure,implementation

如果不使用栈，会很麻烦。
```python
s = input()
while any(s[i] == s[i+1] for i in range(len(s)-1)):
    for j in range(len(s) - 1):
        if s[j] == s[j+1]:
            s = s[:j] + s[j+2:]
            break
if s:
    print(s)
else:
    print("Empty")
```

使用栈：
```python
s = input()
stack = []
for si in s:
    if stack and si == stack[-1]:# 如果有两个重复字母
        stack.pop()# 删除
    else:
        stack.append(si)# 添加字母
if stack:
    for st in stack:
        print(st,end="")
else:
    print("Empty")
```

## E28674:《黑神话：悟空》之加密
http://cs101.openjudge.cn/pctbook/E28674/   
tags:implementation,string,python function

依旧加密题目。
```python
n = int(input())
cypher = input()
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
rolled_alphabet = alphabet[-(n % 26):] + alphabet[:-(n % 26)]
decripted = cypher.translate(
    str.maketrans(alphabet, rolled_alphabet)# 改变大写字母
).translate(
    str.maketrans(alphabet.lower(), rolled_alphabet.lower())# 改变小写字母
)
print(decripted)
```

## E28681:奖学金
http://cs101.openjudge.cn/pctbook/E28681/   
tags: sorting

利用sort()函数的key参数设定排序方式。
```python
n = int(input())
info = []
for _ in range(n):
    data = list(map(int,input().split()))
    info.append([_+1,sum(data),data[0]])#学号，总成绩，语文成绩
info.sort(key = lambda x:100*x[1]+x[2],reverse = True)
# 等价写法：info.sort(key = lambda x:(x[1],x[2]),reverse = True)，注意sort()是稳定排序
for _ in range(min(5,n)):# 输出前五名同学
    print(info[_][0],info[_][1])
```

## E28691:字符串中的整数求和
http://cs101.openjudge.cn/pctbook/E28691/  

哈哈
```python
a=input().split()
print(int(a[0][:2])+int(a[1][:2]))
```

## E29895:分解因数
http://cs101.openjudge.cn/pctbook/E29895/   
tags:math,implementation

试出最小的因数，用自己去除就行了。
```python
n = int(input())
for i in range(2,10**5):
    if n % i == 0:
        print(n // i)
        break
```

## E29940:机器猫斗恶龙
http://cs101.openjudge.cn/pctbook/E29940/   
tags:greedy

假设机器猫0滴血出门，如果扣到负数血量，那么答案就是用它的最低血量的相反数加上1。
```python
n = int(input())
s = 0
m = 0
nums = list(map(int,input().split()))
for i in range(n):
    s += nums[i]
    m = min(m,s)
print(-m+1 if m < 0 else 1)
```
