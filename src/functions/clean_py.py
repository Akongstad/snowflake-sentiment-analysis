
import re

stops = {'we', 'further', "shouldn't", 'won', "that'll", 'from', "hasn't", 'yourselves', 'its', 'shouldn', 'into', 'off', 'it', 'about', 'hasn', 'aren', 'the', "weren't", 'yourself', 'such', 'nor', "don't", 'that', 'm', 'most', 'just', 'some', 'until', 'them', 'what', 'my', 'hers', 'was', 'once', 'both', "needn't", "it's", 'not', "isn't", 'few', 'up', 'himself', 'did', "you've", 'why', 'any', 'below', 'her', 'being', 'didn', 'of', 'between', "you'd", "shan't", 'yours', 'isn', 'your', "you'll", 'he', "wasn't", 'down', "mustn't", 'y', 'd', 'doing', 'in', 'again', 'don', 'were', 'hadn', 'while', 'haven', 'ain', 'more', 'him', 'under', 'against', 'with', 'over', 'by', 's', 'very', 'itself', 'theirs', 'as', 'during', 'wouldn', "mightn't", 're', 'same', 'all', 'than', 'when', 't', 'couldn', 'their', 'how', 'our', 'own', 'for', 'those', 'am', "should've", 'has', 'had', 'i', "won't", 'doesn', 'out', 'through', 'myself', 'will', "aren't", 'ourselves', 'these', "couldn't", 'who', 'weren', 'no', 'or', 'then', "haven't", 'above', "you're", 'so', 'mustn', 'an', 'themselves', 'and', 'there', 'she', 'shan', "wouldn't", 'can', 'herself', 'if', 'where', 'now', "hadn't", 'this', 'mightn', 'his', 'you', 'a', 'they', 'too', 'but', 'to', 'here', 'are', 'ma', 'ours', "she's", 'only', 'needn', "doesn't", 'be', 'll', 'should', 'each', 'at', 've', 'do', 'wasn', 'is', 'me', 'does', 'o', 'before', 'on', 'having', 'other', 'have', "didn't", 'been', 'after', 'because', 'which', 'whom'}
EXAMPLE = "Staying at the Flamingo, our room came with $35 to use in different food places.  Tropical Breeze Cafe is open 24 hours so my friend and me decided to eat here.  Our waitress was kind and friendy as she put up with not nly our drunk table, but other tables too.  $35 got us 2 orders of food, and 2 hot teas.  At 4am, their Patty Melt was greasy and meaty enough to absorb all the alcohol I consumed in the past however many hours.  The only issue I had were the onions.  These onions were cut so wide that each bite was overwhelming.\\n\\nSurprisingly, their thick fries were crispier than I thought."

def clean_py(review:str) -> list[str] :
    # Remove all puctuation and newlines
    cleaned_text = re.sub(r'(\\n)+|[^\w\s$]|[$(\d)]+', ' ', review)
    lower = cleaned_text.lower().split()
    filtered = filter(lambda word: word not in stops, lower)
    return list(filtered)
