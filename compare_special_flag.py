import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import warnings
from sklearn.exceptions import ConvergenceWarning
from collections import Counter

def get_winning_hand(predicted_sazae_hand):
    if predicted_sazae_hand == 'rock':
        return 'paper'
    elif predicted_sazae_hand == 'paper':
        return 'scissors'
    else:
        return 'rock'

def main():
    hands = []
    months = []
    dates = []
    
    with open('sazae_hands.csv', encoding='utf-8') as f:
        next(f)
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 4:
                year, month, day, hand = data[0], data[1], data[2], data[3]
                months.append(int(month))
                hands.append(hand)
                dates.append(f"{year}-{month}-{day}")
                
    date_counts = Counter(dates)
    is_special = [1 if date_counts[d] > 1 else 0 for d in dates]
    
    K_list = list(range(1, 16))
    win_rates_with = []
    win_rates_without = []
    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    print("特番フラグの有無による勝率比較を開始します (max_iter=20固定)...")
    for K in K_list:
        X_with = []
        X_without = []
        y = []
        
        for i in range(K, len(hands)):
            features_base = []
            for j in range(1, K + 1):
                features_base.append(hands[i-j])
            features_base.append(months[i])
            
            X_without.append(features_base.copy())
            features_with = features_base.copy()
            features_with.append(is_special[i])
            X_with.append(features_with)
            y.append(hands[i])
            
        y = np.array(y)
        
        # 特番フラグ「あり」の学習と評価
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_w_encoded = encoder.fit_transform(np.array(X_with))
        X_train, X_test, y_train, y_test = train_test_split(X_w_encoded, y, test_size=0.2, shuffle=False)
        
        model_with = LogisticRegression(solver='saga', max_iter=20, random_state=42)
        model_with.fit(X_train, y_train)
        preds_with = model_with.predict(X_test)
        
        win_count_with = 0
        loss_count_with = 0
        for i in range(len(preds_with)):
            my_hand = get_winning_hand(preds_with[i])
            actual = y_test[i]
            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):
                win_count_with += 1
            elif my_hand != actual:
                loss_count_with += 1
        denominator_with = win_count_with + loss_count_with
        win_rates_with.append((win_count_with / denominator_with * 100) if denominator_with > 0 else 0)
        
        # 特番フラグ「なし」の学習と評価
        X_wo_encoded = encoder.fit_transform(np.array(X_without))
        X_train_wo, X_test_wo, y_train_wo, y_test_wo = train_test_split(X_wo_encoded, y, test_size=0.2, shuffle=False)
        
        model_without = LogisticRegression(solver='saga', max_iter=20, random_state=42)
        model_without.fit(X_train_wo, y_train_wo)
        preds_without = model_without.predict(X_test_wo)
        
        win_count_without = 0
        loss_count_without = 0
        for i in range(len(preds_without)):
            my_hand = get_winning_hand(preds_without[i])
            actual = y_test_wo[i]
            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):
                win_count_without += 1
            elif my_hand != actual:
                loss_count_without += 1
        denominator_without = win_count_without + loss_count_without
        win_rates_without.append((win_count_without / denominator_without * 100) if denominator_without > 0 else 0)

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.plot(K_list, win_rates_with, color='blue', marker='o', linewidth=2.5, label='With Special Flag (特番フラグあり)')
    plt.plot(K_list, win_rates_without, color='red', marker='x', linestyle='--', linewidth=2.5, label='Without Special Flag (特番フラグなし)')
    
    plt.axhline(y=33.33, color='gray', linestyle='--', alpha=0.7, label='Random Chance (33.3%)')
    
    plt.title('Win Rate Comparison: With vs Without Special Program Flag', fontsize=14)
    plt.xlabel('K (Number of past hands included)', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xticks(K_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    output_path = r'C:\Users\kenta\.gemini\antigravity-ide\brain\3fe6c716-e2eb-4b50-9e60-9d04341d1ed7\special_flag_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Graph saved as {output_path}")

if __name__ == '__main__':
    main()
