import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

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
    if hand == "rock":
        return (result_tuple[0] + 1, result_tuple[1], result_tuple[2])
    elif hand == "paper":
        return (result_tuple[0], result_tuple[1] + 1, result_tuple[2])
    else:
        return (result_tuple[0], result_tuple[1], result_tuple[2] + 1)

def learning_rock_paper_scissors(learning_data, learn_during):
    learn_hash = {}
    large_prime = 998244353
    for i in range(learn_during, len(learning_data)):
        next_hand = learning_data[i]
        hash_data = [learning_data[i-j] for j in range(1, learn_during+1)]
        hash_val = hash_convert(hash_data, large_prime)
        res_tuple = (0, 0, 0)
        if hash_val not in learn_hash:
            learn_hash[hash_val] = judge_win_and_tuple_add(res_tuple, next_hand)
        else:
            learn_hash[hash_val] = judge_win_and_tuple_add(learn_hash[hash_val], next_hand)
    return learn_hash

def predict_next_hand(learn_hash, current_state):
    if current_state in learn_hash:
        learn_tuple = learn_hash[current_state]
        if learn_tuple[0] > learn_tuple[1] and learn_tuple[0] > learn_tuple[2]:
            return "paper"
        elif learn_tuple[1] > learn_tuple[0] and learn_tuple[1] > learn_tuple[2]:
            return "scissors"
        else:
            return "rock"
    return "rock"

def judge_correct_detailed(test_raw_hands, learn_during, learn_hash):
    win_count = 0
    loss_count = 0
    win_hands = {'rock': 0, 'paper': 0, 'scissors': 0}
    for i in range(learn_during, len(test_raw_hands)):
        hash_data = [test_raw_hands[i-j] for j in range(1, learn_during+1)]
        hash_val = hash_convert(hash_data, 998244353)
        predicted_hand = predict_next_hand(learn_hash, hash_val)
        
        # 勝ちの判定 (自分が出す手 vs サザエさんの手)
        if (predicted_hand == 'paper' and test_raw_hands[i] == 'rock') or \
           (predicted_hand == 'scissors' and test_raw_hands[i] == 'paper') or \
           (predicted_hand == 'rock' and test_raw_hands[i] == 'scissors'):
            win_count += 1
            win_hands[predicted_hand] += 1
        elif predicted_hand != test_raw_hands[i]:
            loss_count += 1
            
    denominator = win_count + loss_count
    win_rate = win_count / denominator if denominator > 0 else 0
    return win_rate, win_hands

def main():
    hands = []
    # Read the scraped hands
    with open('sazae_hands.csv', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 4:
                hands.append(data[3])

    train_hands, test_hands = train_test_split(hands, test_size=0.2, random_state=42)

    durings = list(range(1, 16))
    win_rates = []
    rock_wins = []
    paper_wins = []
    scissors_wins = []

    for d in durings:
        learning_result = learning_rock_paper_scissors(train_hands, d)
        wr, wh = judge_correct_detailed(test_hands, d, learning_result)
        win_rates.append(wr * 100)
        rock_wins.append(wh['rock'])
        paper_wins.append(wh['paper'])
        scissors_wins.append(wh['scissors'])

    # Create figure with 2 subplots vertically
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1.5, 1]})

    width = 0.6
    
    # --- Plot 1: Absolute Counts and Win Rate ---
    ax1.bar(durings, rock_wins, width, label='Rock Wins', color='#d62728', alpha=0.8)
    ax1.bar(durings, paper_wins, width, bottom=rock_wins, label='Paper Wins', color='#1f77b4', alpha=0.8)
    bottom_scissors = [r + p for r, p in zip(rock_wins, paper_wins)]
    ax1.bar(durings, scissors_wins, width, bottom=bottom_scissors, label='Scissors Wins', color='#2ca02c', alpha=0.8)
    
    ax1.set_ylabel('Number of Wins (Count)', fontsize=12)
    ax1.set_xticks(durings)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_title('Absolute Wins and Win Rate vs. Learn During', fontsize=14)

    # Line chart for win rate on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(durings, win_rates, color='black', marker='D', markersize=6, linewidth=2.5, label='Win Rate (%)')
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.axhline(y=33.33, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Random Chance (33.3%)')
    
    min_rate = min(min(win_rates) - 5, 20)
    max_rate = max(max(win_rates) + 5, 45)
    ax2.set_ylim(min_rate, max_rate)
    ax2.legend(loc='upper right')

    # --- Plot 2: Proportions (100% Stacked Bar) ---
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

    ax3.bar(durings, rock_ratios, width, label='Rock (%)', color='#d62728', alpha=0.8)
    ax3.bar(durings, paper_ratios, width, bottom=rock_ratios, label='Paper (%)', color='#1f77b4', alpha=0.8)
    bottom_scissors_ratios = [r + p for r, p in zip(rock_ratios, paper_ratios)]
    ax3.bar(durings, scissors_ratios, width, bottom=bottom_scissors_ratios, label='Scissors (%)', color='#2ca02c', alpha=0.8)

    ax3.set_xlabel('Learn During (Markov Chain Order)', fontsize=12)
    ax3.set_ylabel('Proportion of Winning Hands (%)', fontsize=12)
    ax3.set_xticks(durings)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_title('Winning Hands Proportion (100% Stacked) vs. Learn During', fontsize=14)

    plt.tight_layout()
    
    # Save the plot
    output_path = r'c:\Users\kenta\pythonPlayGround\marcov_winrate_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Graph saved as {output_path}")

if __name__ == '__main__':
    main()
