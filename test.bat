import random

def compare(yourchoice,computerchoice):
    if not(yourchoice) in {"石头","剪刀","布"}:
        result = '请按照规则出牌'
        return result
    if yourchoice==computerchoice:
        result = '平局'
    elif (yourchoice=="石头" and computerchoice=="剪刀")or(yourchoice=="剪刀" and computerchoice=="布")or(yourchoice=="布" and computerchoice=="石头"):
        result ='您赢了'
    else:
        result='电脑赢了'
    return result

num=random.randint(1,3)
print("猜拳,出！:")
yourguess=input("请出石头、剪刀或布:")
guess={1:"石头",2:"剪刀",3:"布"}
print("电脑出:",guess.get(num))
result=compare(yourguess,guess.get(num))
print(result)
