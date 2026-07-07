import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import warnings
from sklearn.exceptions import ConvergenceWarning

def get_winning_hand(predicted_sazae_hand):
    # サザエさんの予測手に対して、勝てる手を返す
    if predicted_sazae_hand == 'rock':
        return 'paper'
    elif predicted_sazae_hand == 'paper':
        return 'scissors'
    else:
        return 'rock'

def main():
    hands = []
    months = []
    
    with open('sazae_hands.csv', encoding='utf-8') as f:
        next(f) # ヘッダーをスキップ
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 4:
                months.append(int(data[1]))
                hands.append(data[3])
                
    # 特徴量 (Features): [1個前の手, 2個前の手, 放送月]
    # 目的変数 (Target): 現在の手
    X = []
    y = []
    
    for i in range(2, len(hands)):
        hand_t_1 = hands[i-1]
        hand_t_2 = hands[i-2]
        month_t = months[i]
        
        X.append([hand_t_1, hand_t_2, month_t])
        y.append(hands[i])
        
    X = np.array(X)
    y = np.array(y)
    
    # カテゴリ変数のOne-Hotエンコーディング
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    
    # 時系列データなので shuffle=False で分割
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, shuffle=False)
    
    # 学習回数（max_iter）のリスト
    max_iters = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    win_rates = []
    rock_wins = []
    paper_wins = []
    scissors_wins = []
    
    # 収束に関する警告を非表示にする（あえて未収束の反復回数も試すため）
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    print("ロジスティック回帰の学習を開始します...")
    for m_iter in max_iters:
        print(f"max_iter = {m_iter} で学習中...")
        # solver='saga' は確率的勾配降下法に基づくため max_iter の変化が見えやすい
        model = LogisticRegression(solver='saga', max_iter=m_iter, random_state=42)
        model.fit(X_train, y_train)
        
        # テストデータに対して予測
        predictions = model.predict(X_test)
        
        win_count = 0
        loss_count = 0
        wh = {'rock': 0, 'paper': 0, 'scissors': 0}
        
        for i in range(len(predictions)):
            predicted_sazae = predictions[i]
            actual_sazae = y_test[i]
            
            # 予測した手に対して勝てる手を選択
            my_hand = get_winning_hand(predicted_sazae)
            
            # 実際のサザエさんの手と勝敗判定
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

    # グラフの描画
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1.5, 1]})

    x_positions = np.arange(len(max_iters))
    width = 0.6
    
    ax1.bar(x_positions, rock_wins, width, label='Rock Wins', color='#d62728', alpha=0.8)
    ax1.bar(x_positions, paper_wins, width, bottom=rock_wins, label='Paper Wins', color='#1f77b4', alpha=0.8)
    bottom_scissors = [r + p for r, p in zip(rock_wins, paper_wins)]
    ax1.bar(x_positions, scissors_wins, width, bottom=bottom_scissors, label='Scissors Wins', color='#2ca02c', alpha=0.8)
    
    ax1.set_ylabel('Number of Wins (Count)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(max_iters)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_title(f'Logistic Regression: Absolute Wins and Win Rate vs. max_iter', fontsize=14)

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

    ax3.set_xlabel('Max Iterations (Training Iterations)', fontsize=12)
    ax3.set_ylabel('Proportion of Winning Hands (%)', fontsize=12)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(max_iters)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_title('Winning Hands Proportion (100% Stacked) vs. Max Iterations', fontsize=14)

    plt.tight_layout()
    
    output_path = 'logistic_regression_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Graph saved as {output_path}")

if __name__ == '__main__':
    main()
