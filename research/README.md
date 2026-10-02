# Ghi chép nghiên cứu

Các script trong thư mục này chỉ **đọc dữ liệu**, không đặt lệnh. Dữ liệu thô nằm ở `data/` (không đưa lên git).
Mọi con số dưới đây đo trên dữ liệu tới ngày 02/10/2026. Thị trường thay đổi, nên kết quả cần đo lại trước khi dùng.

| Nghiên cứu | Script | Câu hỏi | Kết luận ngắn |
|---|---|---|---|
| [1](#1-đặc-trưng-dự-báo-hướng-vòng-5-phút) | `feature_study.py` | Thông tin nào dự báo được hướng vòng BTC 5 phút? | Chỉ có "giá đã đi so với biến động và thời gian còn lại" (`z`) |
| [2](#2-giá-mua-tối-đa-theo-thời-điểm-vào-lệnh) | `entry_time_study.py` | Vào ở giây t thì thắng bao nhiêu, mua tối đa giá nào? | Giây 60: thắng 54–77% tùy \|z\| |
| [3](#3-độ-trễ-của-odds-so-với-btc) | `odds_lag_study.py` | Odds Predict.fun phản ứng chậm hơn BTC bao lâu? | Khoảng 1 giây, không khai thác được |
| [4](#4-phe-gần-như-chắc-thắng-có-bị-định-giá-quá-cao) | `longshot_study.py` | Phe dẫn đầu ở cuối vòng có bị định giá quá cao? | Không (ở 0,98–0,99); dấu hiệu nhỏ ở 0,80–0,90 |
| [5](#5-phương-pháp-kachoio-trên-btc-5-phút-khóa-lời-khi-giá-dao-động) | `hedge_swing_study.py` | Khóa lời hai phe khi giá dao động trên BTC 5 phút? | Luôn kém hơn giữ tới cuối |
| [6](#6-phương-pháp-kachoio-trên-market-thể-thao-của-predictfun) | `predict_maker_study.py` | Đặt lệnh chờ lệch ≥ 7% so với Polymarket trên market thể thao? | Có lợi thế thật trước trận, nhưng tổng lợi nhuận rất nhỏ |

Chạy lại: `python research/<script>.py --config config.yaml` (xem docstring đầu mỗi file để biết tham số).

---

## Kiến thức nền về sàn

- **Predict.fun** là sàn đứng sau Prediction Market của Binance Wallet. Lệnh thật đi qua CLI `baw` (Binance Agentic Wallet).
- **Phí người khớp chủ động (taker)**, đo trên báo giá thật: phí (tính bằng cổ phiếu) = `2% × min(p, 1−p) / p` số cổ phiếu mua. Bán: `2% × min(p, 1−p) × số cổ phiếu` USDT.
- **Phí người đặt lệnh chờ (maker) = 0**: 0/28.224 lệnh chờ đã khớp phải trả phí.
- Vòng BTC 5 phút chốt theo **Chainlink BTC/USDT**, không theo nến Binance spot. Nến Binance cho kết quả khác kết quả chính thức ở khoảng 1,1% số vòng (3/266).
- Lệnh qua `baw` mất khoảng **12 giây** mới biết khớp hay FAILED.

---

## 1. Đặc trưng dự báo hướng vòng 5 phút

**Dữ liệu:** 14 ngày nến 1 giây Binance (4.030 vòng, 15–29/09).

**Kết quả**
1. **Giá đã đi và thời gian còn lại chi phối tất cả.** Đặt theo hướng giá đã đi thì thắng 65% ở giây 60, 78% ở giây 180 và 89% ở giây 270. Thị trường cũng thấy điều này nên giá đã tính vào.
2. **Biến động quyết định độ chắc chắn.** Biến động 1 giây trước đó dự báo độ lớn của phần còn lại rất mạnh (tương quan +0,46, t = 33). Giờ 13–15 UTC biến động gấp khoảng 2 lần giờ 3–6 UTC. Vì vậy dùng `z = giá đã đi / (σ_1s × √thời gian còn lại)` thay cho ngưỡng bps cố định.
3. **Mô hình bước ngẫu nhiên Φ(z) tự tin quá mức ở hai đầu.** Còn 30 giây, nó nói phe dẫn thắng 99,0%, thực tế đo bằng nến Binance là 96,5% (mỗi nhóm khoảng 1.500 vòng). Vì vậy chiến lược dùng **bảng tỷ lệ thắng đo thực** theo nhóm |z| (`bot/backtest/calibration.py`), không dùng công thức.
4. Động lượng 10–60 giây cuối và dòng lệnh taker 30 giây cuối: tín hiệu yếu (t ≈ 2–3), không đủ tin sau hơn 40 phép thử.
5. Không tác dụng: lợi suất vòng trước, số giao dịch, chế độ biến động, giờ trong ngày (tỷ lệ Up chung 51,0%).

**Cạm bẫy đã gặp:** khối lượng mua chủ động từ đầu vòng ban đầu trông rất mạnh (t đến −6,8) nhưng là hiệu ứng giả của điểm 3; hiệu chỉnh xong thì mất (t từ −1,3 đến −0,2).

## 2. Giá mua tối đa theo thời điểm vào lệnh

**Dữ liệu:** 14 ngày (3.893 vòng), theo hướng giá đã đi ở giây 60, phí 2%. Cột cuối dùng cận dưới 95% của tỷ lệ thắng.

| \|z\| | Vòng | Thắng | Thắng ở phần test (30% cuối) | Giá mua tối đa để hòa vốn |
|---|---|---|---|---|
| 0 – 0,25 | 1.095 | 54,3% | 53,4% | 0,50 |
| 0,25 – 0,5 | 942 | 61,4% | 62,4% | 0,57 |
| 0,5 – 1,0 | 1.235 | 71,6% | 73,8% | 0,68 |
| 1,0 – 1,5 | 467 | 76,7% | 76,1% | 0,71 |
| 1,5 – 2,0 | 119 | 79,0% | 73,0% | 0,69 |
| trên 2,0 | 35 | 100% | 100% | 0,88 (mẫu quá ít) |

**Kiểm tra ngoài mẫu bảng hiệu chỉnh** (hiệu chỉnh 15–25/09, thử 25–29/09):
- Giây 60: bảng đứng vững, sai lệch trong mức nhiễu.
- Giây 270: tỷ lệ thắng thật luôn thấp hơn bảng (|z| > 4: bảng 99,4%, thực tế 96,3% trên 432 vòng, chấm bằng nến Binance). Kết luận lúc đó: **dùng giây 60**.

Từ đây ra chiến lược `value` rồi `momentum_value` (chỉ theo đà, và chỉ khi lời kỳ vọng sau phí thật ≥ 0). Backtest `momentum_value` 30/09–02/10: 180 lệnh, thắng 72,2%, +13,9%/lệnh, sụt tối đa 6,13$, p = 0,026. **Chỉ 2 ngày dữ liệu, trên đúng khoảng dùng để chọn ý tưởng**, nên chưa phải bằng chứng chắc.

## 3. Độ trễ của odds so với BTC

**Dữ liệu:** odds Predict.fun ghi mỗi giây ghép với nến 1 giây Binance (sơ bộ, khoảng 15 phút dữ liệu lúc đo).

- Tương quan mạnh nhất ở **cùng giây** (0,25, t = 7,7), còn một phần nhỏ ở giây sau.
- Sau cú nhảy giá lớn, odds phản ứng khoảng 60% trong cùng giây và gần hết sau 1 giây.
- **Kết luận:** không có độ trễ vài giây để khai thác bằng cách đọc giá mỗi giây. Lệnh qua `baw` mất khoảng 12 giây nên mọi hướng đua tốc độ đều loại.

**Độ chậm có làm mất tiền khi không đua tốc độ?** Đo trên 216 lệnh thật: lệnh đầu khớp lệch +0,0014$/cổ phiếu so với giá lúc quyết định (khoảng 0,2% tiền cược); lệnh thử lại sau FAILED lệch +0,0445 (khoảng 7%); tổng khoảng 0,8%. Chiến lược định giá theo cả vòng 5 phút nên độ chậm gần như không ảnh hưởng.

## 4. Phe "gần như chắc thắng" có bị định giá quá cao?

**Dữ liệu:** 267 vòng BTC 5 phút đã chốt chính thức trên Predict.fun (Chainlink), giá mua thật trên sổ lệnh, phí thật.

1. **Ở giá 0,98–0,99 (15–60 giây cuối): không.** Phe dẫn thắng 100% (15–21 vòng mỗi mốc); mua phe ngược ở 2 xu thua cả 89 lần. Con số 96,3% ở mục 2 là do chấm bằng nến Binance; thị trường đã tính rủi ro lệch nguồn giá vào giá.
2. **Dấu hiệu nhỏ ở giá 0,80–0,90 phút cuối:**

| Giây | Vòng | Giá phe dẫn | Phe dẫn thắng thực tế | Lời nếu mua phe ngược (mỗi 1$) |
|---|---|---|---|---|
| 240 | 33 | 0,861 | 78,8% | +0,33 |
| 255 | 33 | 0,855 | 78,8% | +0,49 |
| 270 | 38 | 0,845 | 63,2% | +1,04 |
| 280 | 30 | 0,848 | 73,3% | +0,38 |
| 285 | 27 | 0,844 | 74,1% | +0,50 |

Chưa tin được: khoảng 30 vòng mỗi mốc, các mốc dùng chung vòng nên không độc lập, và giả thuyết sinh ra từ chính dữ liệu này. Cần kiểm tra **về phía trước** trên dữ liệu mới.

---

## Phương pháp kacho.io

Nguồn: <https://kacho.io/polymarket-arbitrage-real-numbers>

Tác giả chạy bot trên các market esports của Polymarket (CS2, LoL, Valorant, Dota 2), tháng 1–3/2026:
1. Lấy tỷ lệ cược nhà cái (bỏ phần lời của nhà cái) làm **giá công bằng**.
2. **Đặt lệnh chờ (LIMIT)** ở giá rẻ hơn giá công bằng ≥ 7%.
3. Khớp được **cả hai phe** với tổng ≤ 93¢ thì chắc chắn lời; chỉ khớp một phe thì giữ một chiều.

| Kết quả | Số tiền |
|---|---|
| **Lãi ròng** | **+4.973$** trên 95.830$ doanh số (khoảng 5,2%), 3.858 lệnh |
| Cặp khóa được (1.075 cặp) | +8.293$ |
| Lệnh một chiều | −3.185$ (lệnh chờ cũ bị bot nhanh hơn bắt) |
| Trận bị hủy (hoàn 50¢/cổ phiếu) | −134$ |

Lời đến từ bốn điều kiện: (a) có giá công bằng bên ngoài; (b) chênh lệch mua bán rộng (20–30¢); (c) giá công bằng thay đổi chậm (trước trận); (d) làm maker, không đi mua giá người khác. Tác giả tự thừa nhận lời giảm dần khi cạnh tranh tăng và có phí (tháng 3 chỉ còn +390$), và từng phải gánh hơn 3.000$ trên một market.

## 5. Phương pháp kacho.io trên BTC 5 phút: khóa lời khi giá dao động

**So điều kiện:**

| Điều kiện | Esports trong bài | BTC 5 phút trên Predict.fun |
|---|---|---|
| Chênh lệch mua bán | 20–30¢ | trung vị **1¢**; chỉ 0,9% thời gian ≥ 5¢ (175.587 báo giá) |
| Giá công bằng thay đổi | chậm | mỗi giây, theo BTC |
| Ai làm giá | ít bot | bot tạo lập thị trường, phản ứng khoảng 1 giây |
| Tốc độ đặt lệnh | — | 1–12 giây qua `baw` |

Chênh lệch 1¢ nên không có chỗ đặt lệnh chờ lệch 7%. Biến thể còn lại là **khóa lời khi giá dao động**: mua phe theo đà ở giây 60; sau đó nếu phe kia đủ rẻ để `tổng hai phe + phí ≤ 1 − biên` thì mua phe kia (khớp sau 0, 3 hoặc 10 giây); không được thì giữ tới cuối.

**Dữ liệu:** 675 vòng có odds từng giây và kết quả chốt chính thức, phí thật cả hai lệnh.

| Lệnh vào | Giữ tới cuối | Khóa lời, cách tốt nhất trong 12 cách | Chênh lệch |
|---|---|---|---|
| Mọi lệnh `momentum` (502) | **+0,017$/cổ phiếu** | +0,007 (biên 15¢, trễ 10 giây) | −0,010 |
| Chỉ lệnh `momentum_value` (198) | **+0,070$/cổ phiếu** | +0,036 (biên 15¢, trễ 10 giây) | −0,034 |

**Kết luận: ở cả 24 cách thử, khóa lời đều kém giữ tới cuối.** 60–96% số vòng có lúc khóa được, nhưng cơ hội khóa chỉ xuất hiện khi phe mình đang thắng (74–89% vòng khóa được là vòng phe mình rốt cuộc thắng). Khóa lời là đổi một lệnh đang thắng lấy một khoản nhỏ chắc chắn, cộng phí lệnh thứ hai; còn vòng đang thua thì giá không quay về nên vẫn thua trọn.

## 6. Phương pháp kacho.io trên market thể thao của Predict.fun

Predict.fun có market thể thao/esports, và **mọi market đều liên kết với một market Polymarket** (trường `polymarketConditionIds` trong `GET /v1/markets`). Vì vậy giá Polymarket đóng vai trò giá công bằng, thay cho nhà cái trong bài.

### 6.1 Điều kiện

| Điều kiện | Có? | Bằng chứng |
|---|---|---|
| Market thể thao/esports | Có | khoảng 4.200 market thể thao, 45 esports đang mở |
| Giá công bằng bên ngoài | Có | 100% liên kết Polymarket |
| Chênh lệch rộng | Một phần | market lớn (NFL, bóng đá lớn) 1–2¢; market nhỏ: 26% (`SPORTS_MATCH`) và 41% (`SPORTS_TEAM_MATCH`) có chênh lệch ≥ 20¢ |
| Mua ngay (taker) có lời | **Không** | 565 lựa chọn: 1 có lợi thế ≥ 3%, **0 có ≥ 7%**; giá chào bán thường cao hơn Polymarket khoảng 1,5¢ |
| Phí maker | 0 | 0/28.224 lệnh chờ trả phí |

### 6.2 Phương pháp đo

Với mỗi giao dịch quá khứ (`GET /v1/orders/matches`), mỗi phía maker cho biết họ đã mua gì và giá bao nhiêu:
- maker **Bid** phe k ở giá p: maker mua phe k với giá p;
- maker **Ask** phe k ở giá p: maker thực chất mua phe kia với giá 1 − p.

Giá tham chiếu là chuỗi giá Polymarket (`clob.polymarket.com/prices-history`, độ phân giải 1 phút). Lợi thế = `(giá tham chiếu − giá vốn) / giá vốn`. **Markout** là giá Polymarket 5 phút, 30 phút và 2 giờ sau lúc khớp. Audit chặt chỉ dùng giá tham chiếu không cũ quá 120 giây; thực tế 98,2% khối lượng có giá không cũ quá 60 giây.

### 6.3 Kết quả và kiểm định độ bền (800 market, 72 ngày)

Chạy lại audit từ cache bằng `python research/predict_maker_study.py --max-markets 800 --status OPEN`; thêm `--refresh` để lấy lại dữ liệu mạng.

Toàn bộ mẫu có 28.224 maker fill, khoảng 342.000$ (4.746$/ngày). Con số thô ban đầu trông rất tốt: nhóm trước trận, edge 7–50% có 30.572$ khớp và markout 2 giờ +9,2%. Nhưng kết quả này không bền khi kiểm tra lịch thi đấu và mức tập trung.

Bộ lọc audit: còn ít nhất 1 giờ trước trận, giá Polymarket không cũ quá 120 giây, edge từ 7% đến dưới 50%, và ngày trong câu hỏi không lệch `gameStartTime` quá 2 ngày.

| Nhóm sau bộ lọc | Số fill | Market | Khối lượng | $/ngày | Edge lúc khớp | +5 phút | +30 phút | +2 giờ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edge 7–15% | 489 | 102 | 7.290$ | 101,1 | +9,3% | **−4,1%** | −4,7% | **−5,1%** |
| Edge 15–50% | 538 | 78 | 3.047$ | 42,3 | +24,3% | +25,6% | +23,4% | **+20,2%** |
| **Tổng 7–50%** | **1.027** | **134** | **10.338$** | **143,4** | **+13,7%** | +4,6% | +3,6% | **+2,3%** |

Tách theo cấu trúc market cho thấy khác biệt lớn:

| Biến thể | Khối lượng/ngày | Markout 2 giờ |
|---|---:|---:|
| `SPORTS_MATCH` | 64,6$ | **+14,0%** |
| `SPORTS_TEAM_MATCH` | 78,2$ | **−7,4%** |
| Esports | khoảng 0,6$ | mẫu quá nhỏ |

Các phát hiện làm yếu kết luận cũ:

- Bốn market hoãn/đổi lịch chiếm **66,2%** khối lượng ứng viên. Riêng market “FC Cincinnati thắng ngày 05/09”, nhưng `gameStartTime` chuyển sang 21/10, chiếm 62,3% khối lượng trước khi loại. Một fill 18.593$ tạo phần lớn lợi nhuận thống kê.
- Sau khi loại các market đổi lịch, market KBO lớn nhất vẫn chiếm 29,5% khối lượng. Kết quả còn nhạy với vài giao dịch khối.
- Edge 7–15% tổng hợp **không đứng vững**; phần âm tập trung ở `SPORTS_TEAM_MATCH`. Tín hiệu còn lại nằm chủ yếu ở `SPORTS_MATCH` và nhóm edge 15–50%.
- Nhóm edge trên 50% phần lớn là hợp đồng giá rất thấp và hiệu ứng tick, nên loại khỏi audit.
- Trong trận vẫn là bẫy adverse selection; không dùng cho chiến lược này.

### 6.4 Quy mô thực tế

Ước tính cũ “43$/ngày” là mark-to-market của toàn bộ maker và bị giao dịch hoãn trận chi phối, nên không dùng làm kỳ vọng cho bot mới.

Sau bộ lọc, toàn thị trường có 143,4$/ngày khối lượng đã khớp; markout 2 giờ +2,3% tương đương khoảng **3,3$/ngày cho tất cả maker cộng lại**. Bot mới đứng sau hàng đợi giá-thời gian nên chỉ lấy được một phần chưa biết của con số này.

Một trần rất lạc quan, giả sử bot khớp được 1$ ở mọi giao dịch đủ điều kiện, cho 12,8 cơ hội/ngày, 10,3$ khớp/ngày và markout khoảng **+1,70$/ngày**. Đây vẫn chưa phải lợi nhuận chốt thật và giả định hàng đợi không có cạnh tranh.

- Esports gần như không có thanh khoản.
- Vốn có thể bị giữ nhiều ngày tới khi market chốt.
- Với vốn 15$, dữ liệu hiện tại chưa đủ để đưa ra kỳ vọng lợi nhuận đáng tin; nút thắt là xác suất được khớp và rủi ro lịch/settlement.

### 6.5 Kết luận nghiên cứu

1. Có một ứng viên đáng forward-test: maker trên **`SPORTS_MATCH` trước trận** theo giá Polymarket. Chưa có đủ bằng chứng để chạy tiền thật vì quy tắc được chọn trên chính mẫu này, chưa có kết quả chốt và chưa mô hình hóa hàng đợi.
2. Quy tắc paper đề xuất: chỉ khi còn trên 1 giờ; giá tham chiếu ≤ 120 giây; edge 7–50%; loại market đổi lịch trên 2 ngày; giá limit tối đa = `fair / (1 + min_edge)` làm tròn xuống tick; hủy ngay khi edge mất hoặc lịch đổi.
3. Tạm loại `SPORTS_TEAM_MATCH`, mọi market trong trận và edge ≥ 50%. Cược giả lập tối đa 1$ mỗi market để đo đúng bài toán vốn nhỏ.
4. Chỉ xem xét live sau một giai đoạn mới hoàn toàn ngoài mẫu, có fill theo hàng đợi và kết quả settlement thật.

### 6.6 Chưa kiểm chứng

- **Kết quả chốt thật:** markout Polymarket là bằng chứng trung gian, chưa phải PnL cuối cùng. Mẫu `RESOLVED` hiện chưa có fill dùng được.
- **Hàng đợi:** lịch sử chỉ cho biết maker đã đứng sẵn được khớp; chưa cho biết lệnh mới đứng sau bao nhiêu tiền và xác suất khớp.
- **Giá thực thi Polymarket:** `prices-history` là chuỗi giá phút, không phải ảnh chụp bid/ask có kích thước tại đúng thời điểm.
- [Binance Prediction Trading REST API](https://developers.binance.com/en/docs/catalog/web3-wallet-prediction-trading/api/rest-api/trade) có lệnh `LIMIT` với `GTC`, nhưng chưa xác nhận CLI `baw` đang dùng trong repo hỗ trợ đầy đủ đặt, theo dõi và hủy lệnh prediction limit.
- Trận hoãn/hủy và quy tắc settlement có thể tạo chênh lệch hợp lý giữa hai sàn.
- Luật cá cược thể thao nơi người chạy bot sinh sống và điều khoản của sàn.

### 6.7 Bước tiếp theo đề xuất

1. Ghi đồng thời order book Predict.fun, best bid/ask Polymarket, `gameStartTime` và mọi thay đổi lịch.
2. Paper forward-test `SPORTS_MATCH` ít nhất 2–4 tuần; mô phỏng price-time queue và chỉ tính fill khi luồng taker vượt lượng tiền đứng trước.
3. Theo dõi riêng markout 5 phút, 2 giờ và PnL settlement; khóa quy tắc trước khi xem kết quả.
4. Chỉ sau khi paper đạt đủ số market mới thử một limit 1$ để xác nhận phí maker, hành vi partial fill, hủy lệnh và thời gian hoàn vốn.

### Ghi chú kỹ thuật

- `GET https://api.predict.fun/v1/markets?status=OPEN|RESOLVED` (cần `x-api-key`): biến thể `SPORTS_MATCH`, `SPORTS_TEAM_MATCH`, `ESPORTS_*`; trường `polymarketConditionIds`.
- `GET /v1/orders/matches?marketId=…`: `taker`, `makers[]` với `quoteType` (Bid/Ask), `price` và `amount` theo wei (1e18), `fee`, `executedAt`.
- `https://gamma-api.polymarket.com/markets?condition_ids=…`: `clobTokenIds` của từng phe.
- `https://clob.polymarket.com/prices-history?market=<token>&startTs&endTs&fidelity=1`: khoảng dài trả lỗi 400, nên lấy từng cửa sổ ≤ 5 ngày; lỗi thì thử `fidelity=5`.
