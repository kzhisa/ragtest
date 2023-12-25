from bs4 import BeautifulSoup
import argparse

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
vectordb_collection_name = 'wordpress2'

# Embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 2000

# Template
my_template_jp = """あたなは青春18きっぷのエキスパートです。以下の情報を使って、ユーザの質問に日本語で答えてください。回答にあたっては、質問に関係する情報だけを用いて、できるだけわかりやすく回答してください。

情報: {context}

質問: {question}
最終的な回答:"""



'''
不要なタグと冒頭文、関連記事を削除し、残りを<h2>タグ単位で分割し、
文字列の配列として返す
'''
def parse_html_h2(content):

    # bs4 for html parser
    bs_html = BeautifulSoup(content, 'lxml')


    # 図、画像、リンク、テーブルを全て削除
    for tag in bs_html.find_all(['figure', 'img', 'a', 'table']):
        tag.decompose()
    # ブログカード（リンク）を削除
    for tag in bs_html.find_all('div', class_='blogcard'):
        tag.decompose()

    # <h2>の見出し章ごとにテキストを抽出
    sections = []

    # 最初の<h2>タグを検索する
    h2_tag = bs_html.find('h2')

    while h2_tag:
        # セクションのタイトルを取得
        section_title = h2_tag.text
        
        # セクションの内容を取得
        section_content = []
        next_sibling = h2_tag.find_next_sibling()
        
        # 次の<h2>タグまでのすべてのテキストを抽出
        while next_sibling and next_sibling.name != 'h2':
            if next_sibling.string:
                section_content.append(next_sibling.string)
            next_sibling = next_sibling.find_next_sibling()
        
        # セクションをリストに追加
        # 記事最後の「関連記事」のセクションは追加しない
        if r'関連記事' not in section_title:
            sections.append(section_title + '\n' + ''.join(section_content))
        
        # 次の<h2>タグを検索
        h2_tag = next_sibling
    
    return sections


'''
WordpressのXMLファイルから記事を取り出す
記事内容を分割したテキストのリストを返す
'''
def load_xmlfile(xmlfile: str) -> list[str]:

    with open(xmlfile, 'r', encoding="utf-8") as f:

        texts = []

        # BeautifulSoup4でXMLファイルを解析する
        soup = BeautifulSoup(f, "lxml-xml")

        # 記事が格納されている<item>タグを取得
        articles = soup.find_all('item')
        print('Articles: ', len(articles))
        for article in articles:

            # 記事のURL
            url = article.find('link').string

            # 記事のタイトル
            title = article.title.string

            # 記事の内容
            # 記事本体は<content:encoded>にHTMLとして格納されている
            content_html = article.find('content:encoded').get_text()

            # <h2>タグで記事を分割
            splitted_text = parse_html_h2(content_html)

            # documentsに追加
            texts.extend(splitted_text)

    '''
    print('texts: ', len(texts))
    for t in texts:
        #print(len(t))
        print(t)
        print('------------------------')
    '''
    return texts


'''
テキストをEmbeddingしてVectorDBに格納する
作成したVectorDBを永続化する
'''
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


'''
作成したvectorDBを利用して質問に回答する
'''
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

