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
        next(f) # ヘッダーをスキップ
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 4:
                year, month, day, hand = data[0], data[1], data[2], data[3]
                months.append(int(month))
                hands.append(hand)
                dates.append(f"{year}-{month}-{day}")
                
    # 特番かどうかの判定（同じ日付に2回以上データがあれば1、そうでなければ0）
    date_counts = Counter(dates)
    is_special = [1 if date_counts[d] > 1 else 0 for d in dates]
    
    K_list = list(range(1, 16))
    win_rates = []
    rock_wins = []
    paper_wins = []
    scissors_wins = []
    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    print("ロジスティック回帰の学習を開始します (max_iter=20固定)...")
    for K in K_list:
        print(f"K = {K} (過去{K}回分の手 + 放送月 + 特番フラグ) で学習中...")
        
        X = []
        y = []
        
        for i in range(K, len(hands)):
            features = []
            # 1手前〜K手前
            for j in range(1, K + 1):
                features.append(hands[i-j])
            
            # 放送月
            features.append(months[i])
            # 特番フラグ
            features.append(is_special[i])
            
            X.append(features)
            y.append(hands[i])
            
        X = np.array(X)
        y = np.array(y)
        
        # カテゴリ変数のエンコーディング
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)
        
        # 時系列分割
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, shuffle=False)
        
        # max_iterは20で固定
        model = LogisticRegression(solver='saga', max_iter=20, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        win_count = 0
        loss_count = 0
        wh = {'rock': 0, 'paper': 0, 'scissors': 0}
        
        for i in range(len(predictions)):
            predicted_sazae = predictions[i]
            actual_sazae = y_test[i]
            my_hand = get_winning_hand(predicted_sazae)
            
            if (my_hand == 'paper' and actual_sazae == 'rock') or \
               (my_hand == 'scissors' and actual_sazae == 'paper') or \
               (my_hand == 'rock' and actual_sazae == 'scissors'):
                win_count += 1
                wh[my_hand] += 1
            elif my_hand != actual_sazae:
                loss_count += 1
                
        denominator = win_count + loss_count
        wr = win_count / denominator if denominator > 0 else 0
        win_rates.append(wr * 100)
        rock_wins.append(wh['rock'])
        paper_wins.append(wh['paper'])
        scissors_wins.append(wh['scissors'])

    # グラフ描画
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1.5, 1]})

    x_positions = np.arange(len(K_list))
    width = 0.6
    
    ax1.bar(x_positions, rock_wins, width, label='Rock Wins', color='#d62728', alpha=0.8)
    ax1.bar(x_positions, paper_wins, width, bottom=rock_wins, label='Paper Wins', color='#1f77b4', alpha=0.8)
    bottom_scissors = [r + p for r, p in zip(rock_wins, paper_wins)]
    ax1.bar(x_positions, scissors_wins, width, bottom=bottom_scissors, label='Scissors Wins', color='#2ca02c', alpha=0.8)
    
    ax1.set_ylabel('Number of Wins (Count)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(K_list)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_title('Logistic Regression (max_iter=20): Wins and Win Rate vs. K (Past Hands)', fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(x_positions, win_rates, color='black', marker='D', markersize=6, linewidth=2.5, label='Win Rate (%)')
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.axhline(y=33.33, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Random Chance (33.3%)')
    
    min_rate = min(min(win_rates) - 5, 20)
    max_rate = max(max(win_rates) + 5, 45)
    ax2.set_ylim(min_rate, max_rate)
    ax2.legend(loc='upper right')

    rock_ratios = []
    paper_ratios = []
    scissors_ratios = []
    for r, p, s in zip(rock_wins, paper_wins, scissors_wins):
        total_wins = r + p + s
        if total_wins > 0:
            rock_ratios.append(r / total_wins * 100)
            paper_ratios.append(p / total_wins * 100)
            scissors_ratios.append(s / total_wins * 100)
        else:
            rock_ratios.append(0)
            paper_ratios.append(0)
            scissors_ratios.append(0)

    ax3.bar(x_positions, rock_ratios, width, label='Rock (%)', color='#d62728', alpha=0.8)
    ax3.bar(x_positions, paper_ratios, width, bottom=rock_ratios, label='Paper (%)', color='#1f77b4', alpha=0.8)
    bottom_scissors_ratios = [r + p for r, p in zip(rock_ratios, paper_ratios)]
    ax3.bar(x_positions, scissors_ratios, width, bottom=bottom_scissors_ratios, label='Scissors (%)', color='#2ca02c', alpha=0.8)

    ax3.set_xlabel('K (Number of past hands included)', fontsize=12)
    ax3.set_ylabel('Proportion of Winning Hands (%)', fontsize=12)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(K_list)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_title('Winning Hands Proportion (100% Stacked) vs. K', fontsize=14)

    plt.tight_layout()
    
    output_path = 'logistic_regression_k_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Graph saved as {output_path}")

if __name__ == '__main__':
    main()
