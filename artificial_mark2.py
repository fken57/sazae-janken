import numpy as np
from sklearn.model_selection import train_test_split


def learning_rock_paper_scissors(learning_hand_data, learn_during, learning_months_data):
    learn_hash = {}
    large_prime = 998244353
    
    # 最後の要素まで含めるために len(learning_hand_data) に修正
    for i in range(learn_during, len(learning_hand_data)):
        next_hand = learning_hand_data[i]
        next_month = learning_months_data[i]
        hash_data = [learning_hand_data[i-j] for j in range(1, learn_during+1)]
        hash_val = hash_convert(hash_data, large_prime)

        tuple_val = (hash_val, next_month)
            
        res_tuple = (0, 0, 0)
        if tuple_val not in learn_hash:
            learn_hash[tuple_val] = judge_win_and_tuple_add(res_tuple, next_hand)
        else:
            learn_hash[tuple_val] = judge_win_and_tuple_add(learn_hash[tuple_val], next_hand)
            
    # returnをforループの外に出す
    return learn_hash


def hash_convert(hand_data, large_prime):
    hash_val = 0
    for hand in hand_data:
        if hand == "rock":
            hash_val = (hash_val * 31 + 1) % large_prime
        elif hand == "paper":
            hash_val = (hash_val * 31 + 2) % large_prime
        else:
            hash_val = (hash_val * 31 + 3) % large_prime
    return hash_val


def judge_win_and_tuple_add(result_tuple, hand):
    # タプルの要素ごとの足し算になるように修正
    if hand == "rock":
        return (result_tuple[0] + 1, result_tuple[1], result_tuple[2])
    elif hand == "paper":
        return (result_tuple[0], result_tuple[1] + 1, result_tuple[2])
    else:
        return (result_tuple[0], result_tuple[1], result_tuple[2] + 1)
    
    
def predict_next_hand(learn_hash, current_state):
    if current_state in learn_hash:
        learn_tuple = learn_hash[current_state]
        # rock(0)が来そうならpaper、paper(1)ならscissors、scissors(2)ならrockを出す
        if learn_tuple[0] > learn_tuple[1] and learn_tuple[0] > learn_tuple[2]:
            return "paper"
        elif learn_tuple[1] > learn_tuple[0] and learn_tuple[1] > learn_tuple[2]:
            return "scissors"
        else:
            return "rock"
    return "rock"

def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def judge_correct(test_raw_hands, test_months, learn_during, learn_hash):
    win_count = 0
    # 最後の要素まで含める
    for i in range(learn_during, len(test_raw_hands)):
        hash_data = [test_raw_hands[i-j] for j in range(1, learn_during+1)]
        hash_val = hash_convert(hash_data, 998244353)
        predicted_hand = predict_next_hand(learn_hash, (hash_val, test_months[i]))

        # 勝敗判定を正しく修正 (自分が出す手 vs サザエさんの手)
        if (predicted_hand == 'paper' and test_raw_hands[i] == 'rock') or \
           (predicted_hand == 'scissors' and test_raw_hands[i] == 'paper') or \
           (predicted_hand == 'rock' and test_raw_hands[i] == 'scissors'):
            win_count += 1
            
    return win_count / (len(test_raw_hands) - learn_during)


def main():
    hands = []
    months = []
    with open('sazae_hands.csv', encoding='utf-8') as f:
        next(f) # ヘッダー行('year,month,day,sazae_hand')をスキップ
        for line in f:
            data = line.strip().split(',')
            # 空行などを避ける
            if len(data) >= 4:
                hands.append(data[3])
                months.append(int(data[1]))

    train_hands, test_hands = train_test_split(hands, test_size=0.2, random_state=42)
    train_months, test_months = train_test_split(months, test_size=0.2, random_state=42)

    # 過去3回分の手と季節を元に次を予測する (3などの小さな整数を指定する)
    learn_during = 7
    learning_result = learning_rock_paper_scissors(train_hands, learn_during, train_months)
    
    win_rate = judge_correct(test_hands, test_months, learn_during, learning_result)
    print(f"Win rate: {win_rate:.2%}")


if __name__ == '__main__':
    main()