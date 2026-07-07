import urllib.request
from bs4 import BeautifulSoup
import time
import csv

def scrape_sazae_hands(start_year=1996, end_year=2026):
    results = []
    
    for year in range(start_year, end_year + 1):
        url = f"http://park11.wakwak.com/~hkn/result{year}.htm"
        print(f"{year}年のデータを取得中: {url}")
        
        try:
            # 古いサイトなのでShift_JISの可能性が高いと推測
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('shift_jis', errors='replace')
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # テーブル行を取得
            rows = soup.find_all('tr')
            for row in rows:
                cols = row.find_all(['td', 'th'])
                cols_text = [c.get_text(strip=True) for c in cols]
                
                # 想定される列数があり、5列目がじゃんけんの手である行を抽出
                if len(cols_text) >= 5:
                    sazae_hand = cols_text[4]
                    # 有効な手のみを抽出（ヘッダー行などを除外）
                    if sazae_hand in ['グー', 'チョキ', 'パー']:
                        month = cols_text[0]
                        day = cols_text[1]
                        results.append([year, month, day, sazae_hand])
                        
            # サーバーへの負荷を考慮して1秒待機
            time.sleep(1)
            
        except urllib.error.URLError as e:
            print(f"{year}年の取得に失敗しました: {e}")
        except Exception as e:
            print(f"予期せぬエラー ({year}年): {e}")

    return results

hand_dict = {'グー': 'rock', 'チョキ': 'scissors', 'パー': 'paper'}

def main():
    hands = scrape_sazae_hands(1996, 2026)
    
    output_file = 'sazae_hands.csv'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['year', 'month', 'day', 'sazae_hand'])

        hands_in_English = [[hand[0], hand[1], hand[2], hand_dict[hand[3]]] for hand in hands]
        writer.writerows(hands_in_English)
        
    print(f"完了しました！合計 {len(hands)} 件の結果を {output_file} に保存しました。")
    
    # 順番に出した手だけを確認したい場合
    print("\n--- 直近10回分の手 ---")
    for hand in hands[-10:]:
        print(hand[3])

if __name__ == '__main__':
    main()
