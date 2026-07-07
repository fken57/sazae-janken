import os

dir_path = r'c:\Users\kenta\pythonPlayGround'

def replace_in_file(filename, old_str, new_str):
    path = os.path.join(dir_path, filename)
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filename}")
    else:
        print(f"Could not find target in {filename}")

# Fix paths in all scripts
old_path_dir = r"c:\Users\kenta\pythonPlayGround\\"
new_path_dir = r"c:\Users\kenta\pythonPlayGround\\"

for filename in os.listdir(dir_path):
    if filename.endswith(".py"):
        path = os.path.join(dir_path, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if old_path_dir in content:
            content = content.replace(old_path_dir, new_path_dir)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Path updated for {filename}")

# Update analyze_during.py
replace_in_file('analyze_during.py', 
    'total = len(results) - during\n    winrate = win_count / total if total > 0 else 0',
    'loss_count = len(results) - during - win_count - draw_count\n    denominator = win_count + loss_count\n    winrate = win_count / denominator if denominator > 0 else 0')

# Update analyze_during_mark2.py
replace_in_file('analyze_during_mark2.py', 
    'total = len(results) - during\n    winrate = win_count / total if total > 0 else 0',
    'loss_count = len(results) - during - win_count - draw_count\n    denominator = win_count + loss_count\n    winrate = win_count / denominator if denominator > 0 else 0')

# Update q_learning_analysis.py
replace_in_file('q_learning_analysis.py',
    "    win_count = 0\n    win_hands = {'rock': 0, 'paper': 0, 'scissors': 0}",
    "    win_count = 0\n    loss_count = 0\n    win_hands = {'rock': 0, 'paper': 0, 'scissors': 0}")
replace_in_file('q_learning_analysis.py',
    "        if reward == 1:\n            win_count += 1\n            win_hands[action] += 1\n            \n    total = len(test_hands) - learn_during\n    win_rate = win_count / total if total > 0 else 0",
    "        if reward == 1:\n            win_count += 1\n            win_hands[action] += 1\n        elif reward == -1:\n            loss_count += 1\n            \n    denominator = win_count + loss_count\n    win_rate = win_count / denominator if denominator > 0 else 0")

# Update q_learning_analysis_epochs.py
replace_in_file('q_learning_analysis_epochs.py',
    "    win_count = 0\n    win_hands = {'rock': 0, 'paper': 0, 'scissors': 0}",
    "    win_count = 0\n    loss_count = 0\n    win_hands = {'rock': 0, 'paper': 0, 'scissors': 0}")
replace_in_file('q_learning_analysis_epochs.py',
    "        if reward == 1:\n            win_count += 1\n            win_hands[action] += 1\n            \n    total = len(test_hands) - learn_during\n    win_rate = win_count / total if total > 0 else 0",
    "        if reward == 1:\n            win_count += 1\n            win_hands[action] += 1\n        elif reward == -1:\n            loss_count += 1\n            \n    denominator = win_count + loss_count\n    win_rate = win_count / denominator if denominator > 0 else 0")

# Update logistic_regression_analysis.py
replace_in_file('logistic_regression_analysis.py',
    "        win_count = 0\n        wh = {'rock': 0, 'paper': 0, 'scissors': 0}",
    "        win_count = 0\n        loss_count = 0\n        wh = {'rock': 0, 'paper': 0, 'scissors': 0}")
replace_in_file('logistic_regression_analysis.py',
    "            if (my_hand == 'paper' and actual_sazae == 'rock') or \\\n               (my_hand == 'scissors' and actual_sazae == 'paper') or \\\n               (my_hand == 'rock' and actual_sazae == 'scissors'):\n                win_count += 1\n                wh[my_hand] += 1\n                \n        wr = win_count / len(y_test)",
    "            if (my_hand == 'paper' and actual_sazae == 'rock') or \\\n               (my_hand == 'scissors' and actual_sazae == 'paper') or \\\n               (my_hand == 'rock' and actual_sazae == 'scissors'):\n                win_count += 1\n                wh[my_hand] += 1\n            elif my_hand != actual_sazae:\n                loss_count += 1\n                \n        denominator = win_count + loss_count\n        wr = win_count / denominator if denominator > 0 else 0")

# Update logistic_regression_k_dimensions.py
replace_in_file('logistic_regression_k_dimensions.py',
    "        win_count = 0\n        wh = {'rock': 0, 'paper': 0, 'scissors': 0}",
    "        win_count = 0\n        loss_count = 0\n        wh = {'rock': 0, 'paper': 0, 'scissors': 0}")
replace_in_file('logistic_regression_k_dimensions.py',
    "            if (my_hand == 'paper' and actual_sazae == 'rock') or \\\n               (my_hand == 'scissors' and actual_sazae == 'paper') or \\\n               (my_hand == 'rock' and actual_sazae == 'scissors'):\n                win_count += 1\n                wh[my_hand] += 1\n                \n        wr = win_count / len(y_test)",
    "            if (my_hand == 'paper' and actual_sazae == 'rock') or \\\n               (my_hand == 'scissors' and actual_sazae == 'paper') or \\\n               (my_hand == 'rock' and actual_sazae == 'scissors'):\n                win_count += 1\n                wh[my_hand] += 1\n            elif my_hand != actual_sazae:\n                loss_count += 1\n                \n        denominator = win_count + loss_count\n        wr = win_count / denominator if denominator > 0 else 0")

# Update mlp_analysis.py
replace_in_file('mlp_analysis.py',
    "        win_count = 0\n        wh = {'rock': 0, 'paper': 0, 'scissors': 0}",
    "        win_count = 0\n        loss_count = 0\n        wh = {'rock': 0, 'paper': 0, 'scissors': 0}")
replace_in_file('mlp_analysis.py',
    "            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):\n                win_count += 1\n                wh[my_hand] += 1\n                \n        wr = win_count / len(y_test)",
    "            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):\n                win_count += 1\n                wh[my_hand] += 1\n            elif my_hand != actual:\n                loss_count += 1\n                \n        denominator = win_count + loss_count\n        wr = win_count / denominator if denominator > 0 else 0")

# Update compare_special_flag.py
replace_in_file('compare_special_flag.py',
    "        win_count_with = 0\n        for i in range(len(preds_with)):\n            my_hand = get_winning_hand(preds_with[i])\n            actual = y_test[i]\n            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):\n                win_count_with += 1\n        win_rates_with.append(win_count_with / len(y_test) * 100)",
    "        win_count_with = 0\n        loss_count_with = 0\n        for i in range(len(preds_with)):\n            my_hand = get_winning_hand(preds_with[i])\n            actual = y_test[i]\n            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):\n                win_count_with += 1\n            elif my_hand != actual:\n                loss_count_with += 1\n        denominator_with = win_count_with + loss_count_with\n        win_rates_with.append((win_count_with / denominator_with * 100) if denominator_with > 0 else 0)")
replace_in_file('compare_special_flag.py',
    "        win_count_without = 0\n        for i in range(len(preds_without)):\n            my_hand = get_winning_hand(preds_without[i])\n            actual = y_test_wo[i]\n            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):\n                win_count_without += 1\n        win_rates_without.append(win_count_without / len(y_test_wo) * 100)",
    "        win_count_without = 0\n        loss_count_without = 0\n        for i in range(len(preds_without)):\n            my_hand = get_winning_hand(preds_without[i])\n            actual = y_test_wo[i]\n            if (my_hand == 'paper' and actual == 'rock') or (my_hand == 'scissors' and actual == 'paper') or (my_hand == 'rock' and actual == 'scissors'):\n                win_count_without += 1\n            elif my_hand != actual:\n                loss_count_without += 1\n        denominator_without = win_count_without + loss_count_without\n        win_rates_without.append((win_count_without / denominator_without * 100) if denominator_without > 0 else 0)")
