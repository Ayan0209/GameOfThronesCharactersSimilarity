# Game of Thrones Character Analysis

The aim of this project is to analyze the dialogues from the Game of Thrones TV series scripts to identify and visualize the relationships between characters based on the words they use. By narrowing down to the top 25 characters, the project aims to categorize characters with similar personalities based on their dialogue patterns, using machine learning and natural language processing techniques.

## Features of the Code

- **Data Extraction**: Reads and processes a JSON file containing the entire script of the Game of Thrones TV series, extracting dialogues associated with each character.
- **Dialogue Aggregation**: Aggregates the dialogues spoken by each character to create a comprehensive text representation for each character.
- **Text Vectorization**: Utilizes CountVectorizer from the sklearn.feature_extraction.text module to convert the text data into a bag-of-words model, excluding English stop words.
- **Dimensionality Reduction**: Applies t-Distributed Stochastic Neighbor Embedding (TSNE) using sklearn.manifold.TSNE to reduce the high-dimensional text data to a 2D space for visualization.
- **Data Visualization**: Uses plotly.express to create an interactive scatter plot, highlighting the top 25 characters and visualizing their relationships based on the similarity of their dialogue.

## Data Visualization

The scatter plot below visualizes the top 25 characters from the Game of Thrones TV series based on the words they use. Characters that are closer together on the plot are categorized as having similar personalities:

![GOTChar](https://github.com/Ayan0209/GameOfThronesCharactersSimilarity/assets/33597664/370975ef-e337-4448-8a99-06c91ca0021a)
