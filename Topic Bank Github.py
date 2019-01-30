
# coding: utf-8

# # Framing the Problem

# The Canadian banking system continues to rank at the top of the world thanks to our continuous effort to improve our quality control practices. As evident during the 2008 Sub-Prime Mortgage Crisis, Canada was one of the few countries that withstood the Great Recession.
# 
# One approach to improve quality control practices is by analyzing the quality of a Bank's business portfolio for each individual business line. For example, a Bank's core business line can be providing construction loan products, and based on the rationale behind each deal for the approval and denial of construction loans, we can determine the topics in each decision from the rationales. By determining the topics in each decision, we can then perform quality control to ensure all the decisions that were made are in accordance to the Bank's risk appetite and pricing.
# 
# With this approach, the Bank can improve the quality of their construction loan business from their own decision making standards, and thus improving the overall quality of their business.
# 
# However, in order to get this information, the Bank needs to extract topics from hundreds and thousands of data, and then interpret the topics before determining if the decisions that were made meets the Bank's decision making standards, all of which can take a lot of time and resources to complete.
# <br>
# <br>
# 
# **Business Solutions:**
# 
# To solve this issue, I have created a "Quality Control System" that learns and extracts topics from a Bank's rationale for decision making. This can then be used as quality control to determine if the decisions that were made are in accordance to the Bank's standards.
# 
# I will perform a Topic Modeling with Latent Dirichlet Allocation (LDA) Model on an entire department's decision making rationales.
# 
# I will also determine the dominant topic associated to each rationale, as well as determining the rationales for each dominant topics in order to perform quality control analysis.
# 
# <u>Note</u>: Although I will be using real world data from a Bank, however, I will not showcase any relevant information from the actual dataset for privacy protection. Any information shown here will not violate the privacy of the Bank. 
# <br>
# <br>
# 
# **Benefits:**
# - Efficiently determine the main topics of rationale texts in a large dataset
# - Improve the quality control of decisions based on the topics that were extracted
# - Conveniently determine the topics of each rationale
# - Extract detailed information by determining the most relevant rationale for each topic
# <br>
# <br>
# 
# **Robustness:**
# 
# To ensure the model performs well, I will take the following approach:
# - Run the LDA Model and the LDA Mallet (Machine Learning Language Toolkit) Model to compare the performances of each model
# - Run the LDA Mallet Model and optimize the number of topics in the rationale by choosing the optimal model with highest performance
# 
# <u>Note</u> that the main different between LDA Model vs LDA Mallet Model is that LDA Model uses variational Bayes method which is faster but less precise than LDA Mallet Model which uses Gibbs Sampling.
# <br>
# <br>
# 
# **Assumption:**
# - I have taken data from with a sample size of 511, and assuming that this dataset is sufficient to capture the topics in the rationale
# - We're also assuming that the results in this model is applicable in the same way if we were to train an entire population of the rationale dataset with the exception of few parameter tweaks
# <br>
# <br>
# 
# **Future:**
# 
# This model is an innovative way to determine key topics embedded in large quantity of texts, and apply it in a business context to improve a Bank's quality control practices on it's business line. However, since I was not able to fully showcase all the visualizations and output from the results due to privacy protection, please refer to "[Employer Reviews using Topic Modeling](https://stackoverflow.com/questions/43288550/iopub-data-rate-exceeded-in-jupyter-notebook-when-viewing-image)" for more detail.

# # Data Overview

# In[1]:


import pandas as pd
csv = ("audit_rating_banking.csv")
df = pd.read_csv(csv, encoding='latin1') # solves enocding issue when importing csv
df.head(0)


# After importing the data, we see that the "Deal Notes" column is where the rationales are for each deal. This is the column that we are going to use for extracting topics. 
# 
# <u>Note</u> that i did not show any data for privacy protection.

# In[2]:


df = df[['Deal Notes']]
df.shape


# In[3]:


df1 = df.copy()
df1["Deal Notes"] = df1["Deal Notes"].apply(lambda x : x.rsplit(maxsplit=len(x.split())-4)[0]) # sets the character limit to 4 words
df1.loc[2:4, ['Deal Notes']]


# As a expected, we see that there are 511 items in our dataset with 1 data type (text).
# 
# I have also wrote a function showing a sneak peak of the "Rationale" with privacy protection (only the first 4 words are shown).

# # Data Cleansing

# We will use regular expressions to clean our any unfavorable data in our dataset, and then preview what the data looks like after cleansing.

# In[4]:


data = df['Deal Notes'].values.tolist() # convert to list

import re
data = [re.sub(r'[^a-zA-Z ]+', '', sent) for sent in data] # removes everything except letters and space


# In[ ]:


from pprint import pprint
pprint(data[:1])


# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output here are text that has been cleansed with only words and space characters.

# # Pre-Processing

# With our data now cleansed, the next step is to pre-process our data so that it can used for LDA.
# 
# We will perform the following:
# - Breakdown each sentences into a list of words through Tokenization by using Gensim's `simple_preprocess`
# - Additional cleansing by converting text into lowercase while removing punctuations by using Gensim's `simple_preprocess` once again
# - Remove stopwords (words that carry no meaning such as to, the, etc) by using NLTK's `corpus.stopwords`
# - Apply the Bigram and Trigram models for words that occurs together (ie. warrant_proceeding, there_isnt_enough) by using Gensim's `models.phrases.Phraser`
# - Transform words to their root words (ie. walking to walk, mice to mouse) by Lemmatizing the text using `spacy.load(en)` which is Spacy's English dictionary

# ## Tokenization and Additional Cleansing

# In[ ]:


import gensim
from gensim.utils import simple_preprocess 
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
        # deacc=True removes punctuations
        # simple_preprocess to tokenize, and clean up messy text (converts to lowercase and removes punctuations)
        
data_words = list(sent_to_words(data))

print(data_words[:1])   


# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output here are text that broken down into individual words in all lowercases without punctuations.

# ## Remove Stopwords

# In[8]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use']) # add additional stop words

# Define functions for stopwords, bigram, trigram and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    # clean out words using simple_preprocess (gensim) if the words are not already in stop_words (stop_words from NLTK)
    
# Remove Stop Words from simple_preprocess
data_words_nostops = remove_stopwords(data_words)


# ## Create and Apply Trigrams

# In[ ]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# min_count is minimum 5 letters
# threshold is 100 threshold in each word before accepting another word

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_trigram(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Form Trigrams from Gensim.models.phrases
data_words_trigrams = make_trigram(data_words_nostops)


# ## Lemmatize

# In[ ]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): # http://spacy.io/api/annotation
    texts_out = [] # creates a list
    for sent in texts:
        doc = nlp(" ".join(sent)) # adds English dictionary from Spacy to the texts by instantiating doc
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
# adds doc to texts_out list for all base token in doc, if these added doc are from a loose part of the speech in allowed_postags
# in other words, every (base) word that comes in will be added to the list, if these (loose) words are a NOUN, ADJ, VERB, or ADV
# lemma_ is base form of token and pos_ is lose parts of the speech
    return texts_out

# texts -> doc -> token
# texts -> sent -> doc -> token

# Initialize spacy 'en' model, keeping only tagger components (for efficieny)
# python -m spacy download en
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output here are text that are Tokenized, Cleansed, Lemmatized with applicable bigram and trigrams.

# # Prepare Dictionary and Corpus

# Now that are data has been cleansed, and pre-processed, here are the final steps that we need to do before our data is ready for LDA input:
# - Create a dictionary from our pre-processed data using Gensim's `corpora.Dictionary`
# - Create a corpus by applying "term frequency" (word count) to our "pre-processed data dictionary" on the originally pre-processed data using Gensim's `.doc2bow`
#     

# In[11]:


import gensim.corpora as corpora # dictionary

# Create dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create corpus
texts = data_lemmatized

# Term document frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# We can see that our corpus is a list of every word in an index form followed by their count frequency.

# In[12]:


id2word[0]


# We can also see the actual word of each index by calling the index from our pre-processed data dictionary.

# In[ ]:


# Human readable format of corpus (term-frequency) for the first item
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# Lastly, we can see the list of every word in actual word (instead of index form) followed by their count frequency using a simple for loop.
# 
# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output here are a list of text showing words with their corresponding count frequency.
# 
# Now that we have created our dictionary and corpus, we can now feed the data into our LDA Model.

# # LDA Model

# **Latent (hidden) Dirichlet Allocation** is a generative probabilistic model of a documents (composites) made up of words (parts). This is based on the probability of words (parts) when selecting (sampling) topics (category), and the probability of selecting topics (category) when selecting (sampling) a document (composite).
# 
# Essentially we are extracting topics in documents by looking at the probability of words to determine the topics, and then the probability of topics to determine the document. 
# 
# There are two LDA algorithms. The **Variational Bayes** is used by Gensim's **LDA Model** while **Gibb's Sampling** is used by **LDA Mallet Model** using Gensim's Wrappers package.
# 
# Here is the general overview of Variational Bayes and Gibbs Sampling:
# - **Variational Bayes**
#     - sampling the variations between each word (part or variable) to determine which topic it belongs to (but some variations cannot be explained)
#     - Fast but less accurate
# - **Gibb's Sampling (Markov Chain Monte Carlos)**
#     - sampling one variable at a time, conditional upon all other variables
#     - Slow but more accurate

# In[ ]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics = 9, random_state = 100,
                                            update_every = 1, chunksize = 100, passes = 10, alpha = 'auto',
                                            per_word_topics=True)
# (corpus, dictionary, # of topics, random_state, how often the model parameters should be updated, # of document in each training chunk,
# total # of training passes, alpha is hyperparameter that affect sparsity of topics)
# Here we use 9 topics

# Print the keyword the 9 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# After building the LDA Model using Gensim, we display the 9 topics in our document along with the top 10 keywords and their corresponding weights that makes up each topic.
# 
# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output is a list of the 9 topics, and each topic shows the top 10 keywords and their corresponding weights that makes up the topic.

# ## LDA Model Performance

# In[15]:


# Compute perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus)) # A measure of how good the model is (lower the better)

# Compute coherence score
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda) # high the better


# In order to determine the accuracy of the topics that we used, we will compute the Model Perplexity and Coherence Score. The Perplexity score measures how well the LDA predicts the sample (the lower the perplexity score, the better the model predicts). The Coherence score measures the quality of the topics that were learned (the higher the coherence score, the higher the quality of the learned topics).
# 
# Here we see a **Perplexity score of -6.87** (negative due to log space), and **Coherence score of 0.41**. 
# 
# <u>Note</u>: we will only use the Coherence score moving forward as it improve the quality of our learned topics.

# ## Visualize LDA Model

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # hides future warning

import pyLDAvis # interactive visualization for LDA
import pyLDAvis.gensim 

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# We used pyLDAvis to visualize our topics. 
# 
# For interpreting pyLDAvis:
# - Each bubble represent a topic
# - The larger the bubble, the more prevalent is that topic
# - A good topic model has fairly big non-overlapping bubbles scattered through the chart (instead of being clustered in one quadrant)
# - Red highlight: Salient keywords that form the topic (most notable keywords)
# 
# <u>Note</u> that due to privacy protection I have omitted the output.

# # LDA Mallet Model

# Now that we have completed our Topic Model using "Variational Bayes" algorithm from Gensim's LDA, we will now explore Mallet's LDA (which is more accurate but slower) using Gibb's Sampling (Markov Chain Monte Carlos) under Gensim's Wrapper.
# 
# Mallet's LDA model is more accurate it utilizes Gibb's Sampling by sampling one variable at a time conditional upon all other variables.

# In[17]:


# Download file: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
# Installing mallet: https://programminghistorian.org/en/lessons/topic-modeling-and-mallet#mac-instructions
# Download Java SE (download .zip instead of .exe for silent install)

import os
from gensim.models.wrappers import LdaMallet

os.environ.update({'MALLET_HOME':r'/Users/Mick/Desktop/mallet/'}) # set environment
mallet_path = '/Users/Mick/Desktop/mallet/bin/mallet' # update this path


# In[18]:


ldamallet = LdaMallet(mallet_path,corpus=corpus,num_topics=9,id2word=id2word)
# here we use 9 topics again


# In[ ]:


# Show topics
pprint(ldamallet.show_topics(formatted=False))


# After building the LDA Mallet Model using Gensim's Wrapper, we display the 9 topics in our document along with the top 10 keywords and their corresponding weights that makes up each topic.
# 
# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output is a list of the 9 topics, and each topic shows the top 10 keywords and their corresponding weights that makes up the topic.

# ## LDA Mallet Model Performance

# In[20]:


# Compute coherence score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence="c_v")
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# Here we see the Coherence Score for our **LDA Mallet Model** is showing **0.41** which is similar to the LDA Model above. However, given that we are now using a more accurate model from **Gibb's Sampling**, and the purpose of the Coherence Score is to see the quality of the topics that were learned, then our next step is to improve the Coherence Score which will improve the quality of the topics learned.
# 
# To improve the quality of the topics learned, we need to find the optimal number of topics in our document, and once we find the optimal number of topics in our document, then our Coherence Score will be improved as well since all the topics in the documents are extracted accordingly without redundancy.

# # Finding the Optimal Number of Topics for LDA Mallet Model

# We will use the following function to run our **LDA Mallet Model**:
# 
#     compute_coherence_values
#     
# <u>Note</u> that we trained our model to find topics between the range of 2 to 12 topics at an interval of 1.

# In[21]:


# compute coherence_values for LdaMallets models

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model) # adds different LdaMallet models based on num_topics(start, limit, step)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence()) # add the different LdaMallet models and compute Coherence score
    return model_list, coherence_values

# Recall previous LdaMallet model
# ldamallet = LdaMallet(mallet_path,corpus=corpus,num_topics=10,id2word=id2word)
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence="c_v")

# compute a list of LdaMallets models
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=12, step=1)


# Now that the training is complete, the next step is to visualize the results and the display the list of results.

# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# show graph
limit=12; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel('Num Topics')
plt.ylabel('Coherence score')
plt.legend(('coherence_values'), loc='best')
plt.show()


# In[25]:


# Print the coherence scores
for m, cv in zip(x, coherence_values): # zip aggregate iterables (zero or more)
    print('Num Topics =', m, ' has Coherence Value of', round(cv, 4))
    # calls an iterated x (x = range(start, limit, step)) and coherence_values (4 decimals)


# As we can see, the optimal number of topics here is **10 topics** with a Coherence Score of **0.43** which is higher our previous results at 0.41. This also means that there are 10 dominant topics in this document.
# 
# We will proceed to select our final model using 8 topics.

# In[ ]:


# Select the model with highest coherence value and print the topics
optimal_model = model_list[8] # the 8th index from above output
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10)) # set num_words parament to show 10 words per topic


# By using our **Optimal LDA Mallet Model** using Gensim's Wrapper, we display the 10 topics in our document along with the top 10 keywords and their corresponding weights that makes up each topic.
# 
# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output is a list of the 10 topics, and each topic shows the top 10 keywords and their corresponding weights that makes up the topic.

# ## Visual the Optimal LDA Mallet Model

# In[ ]:


# Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 5, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# Here we also visualized the 10 topics in our document along with the top 10 keywords. Each keyword's corresponding weights  are shown by the size of the text.
# 
# <u>Note</u> that due to privacy protection I have omitted the output.

# # Analysis

# Now that our **Optimal Model** is constructed, we will utilize the purpose of LDA and determine the following:
# - Determine the dominant topics for each document
# - Determine the most relevant document for each of the 10 dominant topics
# - Determine the distribution of documents contributed to each of the 10 dominant topics

# ## Finding topics for each document

# In[ ]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data):
    # Create DataFrame
    sent_topics_df = pd.DataFrame()

    # Get dominant topic in each document
    for i, row in enumerate(ldamodel[corpus]):                   # call the optimal_model from LdaMallet (ldamodel=optimal_model)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)    # sort the dominant topic for each document (use row[0] for windows)
        
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:                                           # dominant topic
                wp = ldamodel.show_topic(topic_num)              # show the dominant topic column (topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])   # add keywords column to each dominant topic column
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                # add to dataframe columns: dominant topic (topic_num), perc_contribution (prop_topic (4 decimal)),
                # and keywords (topic_keywords)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'] # create dataframe title
    # Essentially, we sort the dominant topic for each document,
    # Then we add keywords column to each dominant topic column for each document, and append it to our sent_topics_df dataframe
    # (dataframe now include: dominant topic (topic_num), perc_contribution (prop_topic (4 decimal)), and keywords (topic_keywords))
    # Note that prop_topic is perc_contribution from output of keywords and their respective weights (ie. 'window', 0.0235866)
    # Finally name the dataframe columns for topic_num, prop_topic, topic_keywords to Dominant_topic, Perc_Contribution and Topic_Keywords
    
        
    # Add original text to the end of the output (recall that texts = data_lemmatized)
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
    # we take out sent_topics_df dataframe that is configured, and add our texts to it

    
df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
# texts=data since that is what we assigned in the format_topics_sentences function, but in the function we already
# called and added our data (texts=data_lemmatized) to the dataframe, which is why now we use texts=data default parameter


# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()       # re-index
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Document']
# name all the columns for the DataFrame

# Show
df_dominant_topic.head(10)


# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output is a list of the first 10 document with corresponding dominant topics attached.

# ## Finding documents for each topic

# In[ ]:


# Group top 10 documents for the 10 dominant topic
sent_topics_sorteddf_mallet = pd.DataFrame() # Create a new DataFrame for our analysis

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic') 
# Groupyby dominant topics in df_topic_sents_keywords from above (where we derived the DataFrame showing dominant topic for each document)

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)
# Take sent_topics_sorteddf_mallet DataFrame and combine it with,
# the most (head(1)) dominant topic in sent_topcs_outdf_grpd (grp) sort by perc_contribution
# Essentially, we take our dataset with dominant topics and group it with texts that has the most contribution for each dominant topic

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Document"]

# Show the top 10 documents for the 10 dominant topic
sent_topics_sorteddf_mallet 


# <u>Note</u> that due to privacy protection I have omitted the output. However the actual output is a list of most relevant documents for each of the 10 dominant topics

# ## Document distribution across Topics

# In[30]:


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of overall Documents for Each Topic (round to 4 decimals)
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Total Dominant Topic Number
topic_num_keywords = {'Topic_Num': pd.Series([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])}
topic_num_keywords = pd.DataFrame(topic_num_keywords)
                      
# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Reindex
df_dominant_topics.reset_index(drop=True, inplace=True)

# Change Column names
df_dominant_topics.columns = ['Dominant Topic', 'Num_Document', 'Perc_Document']

# Show
df_dominant_topics


# Here we see the number of documents and the percentage of overall documents that contributes to each of the 10 dominant topics.

# # Answering the Questions

# Based on our modeling above, we were able to use a very accurate model from Gibb's Sampling, and further optimize the model by finding the optimal number of dominant topics without redundancy.
# 
# As a result, we are now able to see the 10 dominant topics that were extracted from our dataset. Furthermore, we are also able to see the dominant topic for each of the 511 documents, as well as determining the most relevant document for each dominant topic. 
# 
# With the in-depth analysis of each individual topics and documents above, the Bank can now use this approach as a "Quality Control System" to learn the topics from their rationale in decision making, and then determine if the rationales that were made are in accordance to the Bank's standards for quality control.
