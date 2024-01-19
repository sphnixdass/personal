from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

src_text = [
    """ 
	
	NatWest Group traded 0.9% lower in May to close at 259.6p, outperforming the UK market (FTSE 100 -5.4%, FTSE 250 -3.6%) and trading broadly in line with the wider sector (FTSE Banks -0.5%). Shares traded to their lowest level since November at the start of the month before recovering towards 275p but tracked a broader sell-off to end the month below 260p. Strength in HSBC (+2.9%) and Standard Chartered (+0.8%) offset weakness across the wider sector, with Lloyds closing -8.4%

Equity topics of conversation have been on deposits, valuation, liquidity regulation, competition for funding, Net Interest Margin (NIM) drivers, and, more recently, the focus on asset quality has returned

In Macro news, the International Monetary Fund (IMF) now forecasts that the UK economy will avoid a recession and grow faster than Germany this year after sharply upgrading its views on the back of strong household spending and better relations with the European Union. The Bank of England (BoE) raised interest rates by 25bps to 4.5%, with officials flagging that further increases may be needed if inflationary pressures persist. BoE officials expect inflation to fall to 5.1% in Q4, instead of its previous forecast of 3.9%, but no longer expect the UK economy to fall into a recession

Our average analyst target price is 365p in May, broadly in line with prior month, sell-side analyst recommendations unchanged with 16 buys, six holds and no sells. Of the 22 analysts eight have changed their target price in May (+4 vs the prior month), six downgrades and two upgrades. Earnings Per Share (EPS) forecasts remain broadly in line with April, but with higher forecasts for 2023. The average move in May was +3.0%/-1.0%/-1.9% for 2023/2024/2025 estimates

�1.3bn Directed Buy Back (DBB) takes UK Government holding below 40%: On 22nd May NatWest Group agreed an off-market purchase of 469m shares from UK Government Investments (UKGI), for a total of �1.3bn. This took its stake below 40% for the first time since 2008 (settling at 38.69% of Total Voting Rights �TVR�). UKGI�s trading plan also continued and, combined with the DBB, its NatWest Group stake has reduced further to 38.42% TVR as at 31st May having started the month at 41.59%. As at close on 6th June 2023, 265.2m shares worth �705.4m had been bought back, equivalent to 88.2% of the �800m programme

On Monday 22nd May NatWest Markets Plc successfully launched a CHF 250m five year OpCo transaction, with UBS, Commerzbank and NatWest Markets acting as joint lead bookrunners

Good demand in financial institutions group (FIG) transactions for May, with supply being well absorbed as around �50bn was priced this month in a bounce-back to pre-Silicon Valley Bank levels (approaching February�s circa �53bn). It was interesting to see activity in the Covered Bond market with Nationwide printing its first GBP benchmark transaction of the year, a �750m 5 year SONIA Covered Bond transaction, at SONIA+48bps. The transaction is the tenth GBP SONIA Covered Bond of 2023 and only the fourth from a UK issuer (Santander UK, TSB and Coventry Building Society issued 4-5 year benchmark GBP SONIA Covered Bonds in Q1 2023

Access to the Covered Bond market is a question that is regularly coming up in our investor meetings, as one alternative funding option for banks ahead of TFSME repayments. Given the 4-5 year term nature of Covered Bonds, pricing is obviously attractive relative term deposits, although capacity for supply is more limited (typically around �3bn per annum for the larger UK issuers)

The Credit Rating team provided feedback to Sustainalytics, one of the main Environmental, Social & Governance (ESG) rating agencies, on the NWG and NWM ESG Risk Reports. The NWG ESG Risk Report is expected to be updated by Sustainalytics in Q3 and its important that our feedback is considered during this process

Ratings agency S&P upgraded Barclays Plc to BBB+ based on strong strategic execution, international diversification and cautious balance-sheet management
Moody�s note the credit effects of physical climate risks and natural capital considerations are closely linked. These risks can have significant credits implications by triggering government and regulatory action, greater investor scrutiny and litigation amid changing consumer preferences and technological advancement

Moody�s state that sustained market leadership is central to keeping large banks efficient and profitable. They expect large banks to continue delivering stronger, more stable and more predictable earnings during period of heightened market volatility and macroeconomic uncertainty because of several underlying drivers, including greater diversification.
	
	"""
]

src_text2 = [
	"""I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder """,
	"""I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world. """,
	"""Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share """]

model_name = "D:\\Application\\SCEPOC\\Bert\\"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
#batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)

translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text[0])
print("===============")
print(tgt_text)


#assert (
#    tgt_text[0]
#    == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
#)