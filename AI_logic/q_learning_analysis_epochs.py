import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

ACTIONS = ['rock', 'paper', 'scissors']

def get_reward(ai_action, sazae_action):
    # 勝ち: +1, あいこ: 0, 負け: -1
    if ai_action == sazae_action:
        return 0
    elif (ai_action == 'rock' and sazae_action == 'scissors') or \
         (ai_action == 'paper' and sazae_action == 'rock') or \
         (ai_action == 'scissors' and sazae_action == 'paper'):
        return 1
    else:
        return -1

def get_state(history):
    # 直近のサザエさんの手のリストをタプル化して状態とする
    return tuple(history)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}
        return self.q_table[state][action]
        
    def choose_action(self, state, test_mode=False):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}
            
        # 学習時は epsilon-greedy
        if not test_mode and random.random() < self.epsilon:
            return random.choice(ACTIONS)
            
        # テスト時は完全な貪欲法(greedy)
        max_q = max(self.q_table[state].values())
        best_actions = [a for a, q in self.q_table[state].items() if q == max_q]
        return random.choice(best_actions)
        
    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ACTIONS}
            
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.get_q_value(state, action)
        
        # Q値の更新式
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

def train_q_learning(train_hands, learn_during, epochs=100):
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    for _ in range(epochs):
        for i in range(learn_during, len(train_hands)):
            # 状態は (i - learn_during) から i - 1 番目までの手
            state = get_state(train_hands[i-learn_during:i])
            sazae_hand = train_hands[i]
            
            # 状態から行動を選択
            action = agent.choose_action(state, test_mode=False)
            
            # 行動に対する報酬を観測
            reward = get_reward(action, sazae_hand)
            
            # 次の状態を観測
            if i + 1 < len(train_hands):
                next_state = get_state(train_hands[i-learn_during+1:i+1])
            else:
                next_state = state # 終端処理の簡易化
                
            # Q値を更新
            agent.update(state, action, reward, next_state)
            
    return agent

def test_q_learning(test_hands, learn_during, agent):
    win_count = 0
    loss_count = 0
    win_hands = {'rock': 0, 'paper': 0, 'scissors': 0}
    
    for i in range(learn_during, len(test_hands)):
        state = get_state(test_hands[i-learn_during:i])
        sazae_hand = test_hands[i]
        
        # テスト時はテストモードをTrueにして探索させない
        action = agent.choose_action(state, test_mode=True)
        reward = get_reward(action, sazae_hand)
        
        if reward == 1:
            win_count += 1
            win_hands[action] += 1
        elif reward == -1:
            loss_count += 1
            
    denominator = win_count + loss_count
    win_rate = win_count / denominator if denominator > 0 else 0
    return win_rate, win_hands

def main():
    random.seed(42)
    hands = []
    with open('sazae_hands.csv', encoding='utf-8') as f:
        next(f) # ヘッダーをスキップ
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 4:
                hands.append(data[3])

    train_hands, test_hands = train_test_split(hands, test_size=0.2, random_state=42)

    # 調べるエポック数のリスト
    epochs_list = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500,1000,2000,5000]
    fixed_during = 3
    
    win_rates = []
    rock_wins = []
    paper_wins = []
    scissors_wins = []

    print(f"Q学習のトレーニングを開始します (learn_during = {fixed_during})...")
    for e in epochs_list:
        print(f"epochs = {e} で学習中...")
        agent = train_q_learning(train_hands, fixed_during, epochs=e)
        wr, wh = test_q_learning(test_hands, fixed_during, agent)
        win_rates.append(wr * 100)
        rock_wins.append(wh['rock'])
        paper_wins.append(wh['paper'])
        scissors_wins.append(wh['scissors'])

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1.5, 1]})

    x_positions = np.arange(len(epochs_list))
    width = 0.6
    
    ax1.bar(x_positions, rock_wins, width, label='Rock Wins', color='#d62728', alpha=0.8)
    ax1.bar(x_positions, paper_wins, width, bottom=rock_wins, label='Paper Wins', color='#1f77b4', alpha=0.8)
    bottom_scissors = [r + p for r, p in zip(rock_wins, paper_wins)]
    ax1.bar(x_positions, scissors_wins, width, bottom=bottom_scissors, label='Scissors Wins', color='#2ca02c', alpha=0.8)
    
    ax1.set_ylabel('Number of Wins (Count)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(epochs_list)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_title(f'Q-Learning (learn_during={fixed_during}): Absolute Wins and Win Rate vs. Epochs', fontsize=14)

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

    ax3.set_xlabel('Epochs (Training Iterations)', fontsize=12)
    ax3.set_ylabel('Proportion of Winning Hands (%)', fontsize=12)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(epochs_list)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_title('Winning Hands Proportion (100% Stacked) vs. Epochs', fontsize=14)

    plt.tight_layout()
    
    output_path = 'q_learning_epochs_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Graph saved as {output_path}")

if __name__ == '__main__':
    main()
