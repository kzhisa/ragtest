from bs4 import BeautifulSoup
import argparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
import chromadb

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--make_index', help='Make index from Wordpress XML file')
parser.add_argument('-q', '--query', help='Query')

# Chroma DB
persist_directory = './sampledb'
vectordb_collection_name = 'wordpress'

# Embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 2000

# Template
my_template_jp = """あたなは青春18きっぷのエキスパートです。以下の情報を使って、ユーザの質問に日本語で答えてください。回答にあたっては、質問に関係する情報だけを用いて、できるだけわかりやすく回答してください。

情報: {context}

質問: {question}
最終的な回答:"""


# WordpressのXMLファイルから記事を取り出す
# 記事内容を分割したテキストのリストを返す
def load_xmlfile(xmlfile: str) -> list[str]:

    with open(xmlfile, 'r', encoding="utf-8") as f:

        texts = []

        # BeautifulSoup4でXMLファイルを解析する
        soup = BeautifulSoup(f, "lxml-xml")

        # 記事が格納されている<item>タグを取得
        articles = soup.find_all('item')
        print('Articles: ', len(articles))
        for article in articles:

            # 記事の内容
            # 記事本体は<content:encoded>にHTMLとして格納されている
            content_html = article.find('content:encoded').get_text()

            # 記事本体をからテキストのみを抽出
            content_soup = BeautifulSoup(content_html, 'lxml')
            content_text = content_soup.get_text()

            # テキストをChunkに分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = CHUNK_SIZE,
                chunk_overlap  = 0,
                length_function = len,
            )
            splitted_text = text_splitter.split_text(content_text)

            # documentsに追加
            texts.extend(splitted_text)

    return texts


# テキストをEmbeddingしてVectorDBに格納する
# 作成したVectorDBを永続化する
def create_vectoredb(texts: list[str]):

    # embedding model を設定（OpenAIのtext-embedding-ada-002を利用）
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # path=persist_directoryを指定するとファイルとしてDB内容が永続化される
    client = chromadb.PersistentClient(path=persist_directory)

    # Vectore store としてChromaDBを作成
    vectordb = Chroma(
        collection_name=vectordb_collection_name,
        embedding_function=embeddings,
        client=client,
    )

    # textsを一つずつEmbeddingしてChromaDBに格納する
    # ここでOpenAI API経由でemneddingsし、
    # text本体とともにvectordbに格納される
    vectordb.add_texts(texts)


# 作成したvectorDBを利用して質問に回答する
def query(query: str):

    # chrome db
    embeddings = OpenAIEmbeddings()
    client = chromadb.PersistentClient(path=persist_directory)
    vectordb = Chroma(
        collection_name=vectordb_collection_name,
        embedding_function=embeddings,
        client=client,
    )

    # retriever
    # VectorDBからクエリに関係するテキストを検索するretrieverを設定
    # search_type="simirality"はコサイン類似度による基本的な検索
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"K": 5}
    )

    # model
    # クエリに利用するOpenAIのモデルを設定
    model = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-1106")

    # prompt
    # プロンプトを設定
    prompt = PromptTemplate(
                template=my_template_jp,
                input_variables=["context", "question"],
            )

    # output paeser
    # LLMの出力を文字列として返すパーサーを設定
    output_parser = StrOutputParser()

    # Query Chain
    # LangchainのLCEL記法でChainを設定
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )

    # Chainを実行
    result = chain.invoke(query)

    # 結果を表示
    print('回答:\n', result)


# main
if __name__ == '__main__':

    # Args
    args = parser.parse_args()

    if args.make_index:
        texts = load_xmlfile(args.make_index)
        create_vectoredb(texts)
    
    elif args.query:
        query(args.query)

