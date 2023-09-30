import streamlit as st
import pandas as pd
from textblob import TextBlob
from redlines import Redlines
import re
import string
import heapq
import altair as alt
from nltk.corpus import stopwords
from collections import defaultdict

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
 
def get_summary(text:str,no:int):
    
    rmvwhitespace=re.sub(pattern=r'\s',repl=' ',string=text)
    punct=f'[{string.punctuation}]'
    rmvpunct=re.sub(pattern=punct,repl=' ',string=rmvwhitespace)
    rmvpunct=rmvpunct.lower()
    token=TextBlob(rmvpunct)
    stop_words=stopwords.words('english')
    unique_tokens=set(token.words) - set(stop_words)
    unique_tokens_freq ={}
    for tok in unique_tokens:
        unique_tokens_freq[tok]=token.word_counts[tok] 
    w=max(unique_tokens_freq.values())
    unique_tokens_freq_scaled={}
    for k, v in unique_tokens_freq.items():
        unique_tokens_freq_scaled[k]=v/w
    sentence_score=defaultdict(lambda:0)
    for sentence in TextBlob(text).sentences:
        for word in sentence.words:
            try:
                score=unique_tokens_freq_scaled[word.lower()]
            except:
                score=0 
            sentence_score[str(sentence)]  += score
 
    best_sentence=heapq.nlargest(n=no,iterable=sentence_score,key=sentence_score.get)
    summary='\n\n'.join(best_sentence)
    token_df=pd.DataFrame(data={"Words":unique_tokens_freq.keys(),"Count":unique_tokens_freq.values()})

    return summary ,token_df


def get_pos(text: str):
    blob = TextBlob(text)
    tags = {}
    for z, n in blob.tags:
        tags[n] = z

    df = pd.DataFrame(data=tags, index=['words'])

    return df


def get_sentiment(text: str, threshold: float = 0.3):
    blob = TextBlob(text)
    sentiment: float = blob.sentiment.polarity
    friendly_threshold: float = threshold
    hostile_threshold: float = -threshold

    if sentiment >= friendly_threshold:
        return ('😍😘😍❤', sentiment)
    elif sentiment <= hostile_threshold:
        return ('😢😢😢😢', sentiment)
    else:
        return ('😶😑😐🤨', sentiment)


def get_sentimental_sen(text: str, threshold: float = 0.3):
    """This function is used to split the sentence and get sentiments"""
    blob = TextBlob(text.strip())
    result = []
    for sen in blob.sentences:
        res = " "
        p = sen.sentiment.polarity
        if p >= threshold:
            res = "😄"
        elif p <= -threshold:
            res = "😢"
        else:
            res = "😐"
        res = str(sen)+":"+res
        result.append(res)

    return result


def spell_check(text: str):
    spellcheck = TextBlob(text)
    correct=spellcheck.correct()
    mask=[a!=b for a,b in zip(text.split(), list(correct.split()))]
    definiton={}
    for i,word in enumerate(correct.words):
        if mask[i]:
            definiton[str(word)]=word.define()
    return correct,definiton




caption_block =""":grey[Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language. Organizations today have large volumes of voice and text data from various communication channels like emails, text messages, social media newsfeeds, video, audio, and more. They use NLP software to automatically process this data, analyze the intent or sentiment in the message, and respond in real time to human communication.]"""


def red(spellcheck, spellcheck_output):
    compares = Redlines(source=spellcheck, test=spellcheck_output).compare()
    return compares


st.title('Natural Language Processing')
st.caption(caption_block)
tab1, tab2, tab3,tab4 = st.tabs(tabs=[':blue[Sentiment Analysis]', ':red[Spelling Correction]',':violet[Part of Speech ]',':orange[Text Summarization]'])
with tab1:
    commentry = """Coleman had to repeatedly say his catch phrase for it to stick, but Wolstenholme only had to say his most memorable phrase once.

With England 3-2 up in extra time of the World Cup final, striker Geoff Hurst charged up the field on the break in the dying moments.

Some of the over-excited Wembley crowd invaded the field in the mistaken belief that the final whistle had been blown. As Hurst smashed the ball past West Germany goalkeeper Hans Tilkowski, Wolstenholme uttered the immortal words, "Here comes Hurst. Some people are on the pitch, they think it's all over. It is now! It's four!"

Wolstenholme's words have taken on extra significance as the years have passed with no more English success on the international stage, and there was even a sports quiz show which took the famous line for its name in the 1990's"""
    st.header("Sentimental Analysis")
    sentiment_text = st.text_area(label="Enter the to analyse", value=commentry)
    sentiment_output = get_sentiment(text=sentiment_text)
    st.metric(label="Score", value=sentiment_output[0], delta=round(
        sentiment_output[1], 2))
    sentence_sentiment = get_sentimental_sen(sentiment_text)
    for x in sentence_sentiment:

        st.markdown(x)

with tab2:
    st.header("Spelling Check")
    spellcheck = st.text_input(
        "Enter the text to spell check", value="i made sume istake")
    spellcheck_output,definition = spell_check(text=spellcheck)
    redcorrection = red(str(spellcheck), str(spellcheck_output))
    st.write(spellcheck_output)
    st.caption(redcorrection, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Input📝")
        st.write(spellcheck)
    with col2:
        st.markdown("Corrected ✅✨")
        st.write(spellcheck_output)

    for k,v in definition.items():
        with st.expander(label=k,expanded=True):
            for exp in v: 
                st.markdown('👉'+str(exp))
            


with tab3:
    st.header("Part of speech Tagging")
    pos = st.text_input(label="Enter the Text:", value="I am on the phone")
    pos_output = get_pos(pos)
    st.caption("Part of speech Tags")
    st.write(pos_output)

with tab4:
    st.header("Text Summarization")
   
    summary_input=st.text_area(label="Enter the text to summarize",height=400, value="""Cats are of three types- house cats, farm cats and feral cats. House cats are the cats we pet in our houses. Cats become good friends of humans. Unlike dogs, cats are not very active around their owners. However, they are good emotional companions to their owners. An essay on cats must emphasize the fact that cat-sitting has been proven to be therapeutic by many researchers. 

 

Any ‘my pet cat essay for Class 6’ must include a few details about the appearance of cats. Cats have very sweet features. It has two beautiful eyes, adorably tiny paws, sharp claws, and two perky ears which are very sensitive to sounds. It has a tiny body covered with smooth fur and it has a furry tail as well. Cats have an adorable face with a tiny nose, a big mouth and a few whiskers under its nose. Cats are generally white in colour but can also be brown, black, grey, cream or buff. 

 

Cats are omnivores. They eat vegetative items such as rice, milk, pulses, etc. as well as fish, meat, birds, mice, etc. Therefore, cats can feed on both types of food.

 

It is worth mentioning in this my pet cat essay for Class 6 that cats are considered sacred in several cultures such as the Japanese culture. Cats are often depicted as symbols of wit and honour. Several folklores include stories about the intelligence of cats. 

 

Apart from being clever and sweet, cats are also skilful hunters. They use their sharp, pointed nails and canines (teeth) to kill animals like snakes, mice and also small birds. Cats are also helpful to their owners as they protect the household from rats. Thus, from this cat essay, it can be said that cats are helpful pets as well.

 

However, any essay on cats would be incomplete without writing about their babies. A cat offspring is called a “kitten”. Cats are very protective and caring towards their kittens. They feed the kittens and raise them. Kittens are extremely tiny and adorable as well. Their eyes open sometime after they are born. Kittens are very energetic and they spend their time playing with each other and loving their parents. 

 

Now this cat essay will discuss the nature of cats. Cats are very lazy creatures. They usually spend their time napping and sleeping in warm places. Cats have a slow approach to their lives. They are not very energetic animals and they yawn very adorably whenever they are tired. Cats are very good friends to humans if they trust them. Cats like to sleep close to humans for their body warmth.""")
    number=st.number_input(label='Select how many sentences do you want in the summary',min_value=1,value=3)
    summary_output,df_output=get_summary(text=summary_input,no=number)
    st.write(summary_output)
    st.divider()
    token_chart=alt.Chart(data=df_output).mark_circle().encode(x="Words",y="Count",size="Count")
    st.altair_chart(altair_chart=token_chart,use_container_width=True)


