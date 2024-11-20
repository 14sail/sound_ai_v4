# sound_ai_v4

## data preprocessing 
* 先前以確認且處理過Rick的資料集（英文help與alarm）
* other 從yt找了多組的不同場域的聲音和真實人閒聊的影片音訊擷取 隨機採剪而成
* 這隻程式碼將進行資料的擴增

* This code is mainly based on Rick's fold 1 clean and check his data is okay.
* I found some problems in the v1. For example :the raw data is not the real raw, and some audio has a different preprocessing.

* We have a new target, so we need to make a new corrected data, and also need to make the automatic flowork data building and analysis. (if have time the EDA is also important)

**<font color=#808080> == notes == </font>**

| 類別               | 數量 (原始)      | 數量 (擴增後的訓練集) | 數量 (未擴增驗證集)(只有padding)      |
|--------------------|----------------|--------------------|----------------------- |
| 0: Environment     | 6066           | 12125              | 218                    |
| 1: en_help         | 140            | 420                | 38                     |
| 2: ch_help         | 412            | 1235               | 50                     |
| 3: ja_help         | 79             | 237                | 30                     |
| 4: tw_help         | 238            | 714                | 42                     |
| 5: dog             | 517            | 1550               | 55                     |
| 6: cat             | 219            | 656                | 52                     |
| 7: flush           | 207            | 621                | 42                     |
| 8: alarm           | 201            | 402                | 20                     |
| 9: glass_breaking  | 140            | 419                | 38                     |
