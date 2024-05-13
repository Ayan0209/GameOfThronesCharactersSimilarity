import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import plotly.express as px

df = pd.read_json('script-bag-of-words.json')

#print(df.tail(5))

dialouge = {}

for index, row in df.iterrows():
    for item in row['text']:
        if item['name'] in dialouge:
            dialouge[item['name']] = dialouge[item['name']] + item['text']
        else:
            dialouge[item['name']] = item['text'] + " "
            
print(len(dialouge))

new_df = pd.DataFrame()
new_df['character'] = dialouge.keys()
new_df['words'] = dialouge.values()

print(new_df.iloc[:,0:3].head())

new_df['num_words'] = new_df['words'].apply(lambda x: len(x.split()))
new_df = new_df.sort_values('num_words',ascending=False)
new_df = new_df.head(100)
print(new_df.shape)

cv = CountVectorizer(stop_words='english')
embedding = cv.fit_transform(new_df['words']).toarray()
print(embedding.shape)

embedding = embedding.astype('float64')
tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(embedding)

print(z.shape)

new_df['x'] = z.T[0]
new_df['y'] = z.T[1]


fig = px.scatter(new_df.head(25), x="x", y="y", color="character")
fig.show()

new_df.to_csv('character_embeddings.csv', index=False)
