from bs4 import BeautifulSoup, NavigableString
import argparse
import re
import uuid

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.globals import set_debug, set_verbose
from langchain.callbacks.tracers import ConsoleCallbackHandler
import chromadb

import pickle
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Debug, Verbose
set_verbose(True)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--make_index', help='Make index from Wordpress XML file')
parser.add_argument('-q', '--query', help='Query')
parser.add_argument('-r', '--retrieve', help='Retrieve')

# Chroma DB, Docstore
persist_directory = './sampledb3'
vectordb_collection_name = 'wordpress'
docstore_filename = './wordpress_docstore.bin'
 
# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo-1106"
#LLM_MODEL = "gpt-4-1106-preview"

# TOP_K for retriever
TOP_K = 10

# Langchain debug
#set_debug(True)
#set_verbose(True)

# Template
my_template_jp = """あたなは青春18きっぷのエキスパートです。以下の情報を使って、ユーザの質問に日本語で答えてください。回答にあたっては、質問に関係する情報だけを用いて、できるだけわかりやすく回答してください。

情報: {context}

質問: {question}
最終的な回答:"""


'''
HTMLを<h2>タグ単位で分割する
不要なタグ(figure, img, a, table)を削除する
分割した配列を返す
'''
def split_html_by_h2(content: str) -> list[str]:

    # Wordpress captionのショートコードを削除
    updated_content = re.sub(r'\[caption[^\]]*\].*?\[/caption\]', '', content)

    # bs4 for html parser
    soup = BeautifulSoup(updated_content, 'lxml')

    # 図、画像、リンク、テーブルを全て削除
    for tag in soup.find_all(['figure', 'img', 'a', 'table', 'script']):
        tag.decompose()
    # ブログカード（リンク）を削除
    for tag in soup.find_all('div', class_='blogcard'):
        tag.decompose()


    # <h2>の見出し章ごとにテキストを抽出
    sections = []

    # 最初の<h2>タグを検索する
    h2_tag = soup.find('h2')

    while h2_tag:
        # <h2>タグ
        section_tag = h2_tag
        
        # セクションの内容を取得
        section_content = []
        next_sibling = h2_tag.find_next_sibling()
        
        # 次の<h2>タグまでのすべてのHTMLを追加
        while next_sibling and next_sibling.name != 'h2':
            if next_sibling:
                section_content.append(next_sibling)
            next_sibling = next_sibling.find_next_sibling()
        
        # セクションをリストに追加
        # 記事最後の「関連記事」のセクションは追加しない
        if r'関連記事' not in section_tag.text:
            sections.append(str(section_tag) + '\n' + ''.join([ str(x) for x in section_content ]))
        
        # 次の<h2>タグを検索
        h2_tag = next_sibling

    return sections


'''
HTMLを解析・適切に分割してChunkを作成
 - docs: <h2>タグ単位のChunk
 - sud_docs: <h3>単位または長い<h2>を分割したChunk（vector store 用）
 - metadatas: doc_id（vector store 用）
'''
def create_docs_from_html(contents: list[str]):

    docs = []
    doc_ids = []
    sub_docs = []
    metadatas = []

    for content in contents:

        # bs4 for html parser
        soup = BeautifulSoup(content, 'lxml')

        # docs
        doc = soup.get_text()
        docs.append(doc)
        uid = str(uuid.uuid4())
        doc_ids.append(uid)

        texts = []

        # <h3>タグを全て見つける
        tags = soup.find_all(['h2', 'h3'])

        for tag in tags:

            # セクションのテキストを取得
            tmp = [tag.text + '\n']
            for sibling in tag.next_siblings:
                if sibling.name == 'h3':
                    break
                if isinstance(sibling, NavigableString):
                    tmp.append(str(sibling).strip())
                elif sibling.name:
                    tmp.append(sibling.get_text(strip=True, separator="\n"))
            
            # <h3>セクションのテキストを追加
            texts.append(''.join(tmp))
        
        sub_docs.extend(texts)
        metadatas += [{'doc_id': uid} for _ in texts]

    return {'docs': docs,
            'doc_ids': doc_ids,
            'sub_docs': sub_docs,
            'metadatas': metadatas}



'''
WordpressのXMLファイルから記事のHTMLを取り出す
<h2>タグに分割した記事のリストを返す
'''
def load_xmlfile(xmlfile: str) -> list[str]:

    with open(xmlfile, 'r', encoding="utf-8") as f:

        sections = []

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

            # <h2>タグでHTMLを分割y
            splitted_html = split_html_by_h2(content_html)

            # documentsに追加
            sections.extend(splitted_html)

    return sections


'''
sub_docsからVectore Store(ChromaDB)を、
docsからdocstoreを作成
'''
def create_store(data: dict):

    # Chroma Vectore Store を生成
    
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

    # sub_docsとmetadataをChromaDBに格納する
    vectordb.add_texts(data['sub_docs'], data['metadatas'])

    # Docstoreを生成
    docstore = InMemoryStore()
    docstore.mset(list(zip(data['doc_ids'], data['docs'])))

    # DocstoreをPickle形式で保存
    pickle.dump(docstore, open(docstore_filename, 'wb'))



'''
MultivectorRetrieverを作成
'''
def create_retriever():

    # chrome db
    embeddings = OpenAIEmbeddings()
    client = chromadb.PersistentClient(path=persist_directory)
    vectordb = Chroma(
        collection_name=vectordb_collection_name,
        embedding_function=embeddings,
        client=client,
    )

    # docstoreをpickleから読み込む
    docstore = pickle.load(open(docstore_filename, 'rb'))

    # MultivectoreRetrieverを設定
    # vectorestoreにはChromaで作成したVectore Storeを、
    # docstoreにはInMemoryStoreで作成したDocstoreを指定
    retriever = MultiVectorRetriever(
        vectorstore=vectordb, 
        docstore=docstore,
        id_key='doc_id',
        search_kwargs={"K": TOP_K},
    )

    return retriever


'''
MultivectorRetrieverでドキュメントを検索
'''
def retrieve(query: str):

    # Multivector Retriever
    retriever = create_retriever()

    # retrieve
    result = retriever.get_relevant_documents(query)

    for item in result:
        print('-----')
        print(item)


'''
作成したvector Store と Docstore を用いて、
MultivectorRetrieverで質問に回答する
'''
def query(query: str):

    # Multivector Retriever
    retriever = create_retriever()

    # model
    # クエリに利用するOpenAIのモデルを設定
    model = ChatOpenAI(
        temperature=0,
        model_name=LLM_MODEL)

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
    #result = chain.invoke(query, config={'callbacks': [ConsoleCallbackHandler()]})
    result = chain.invoke(query)

    # 結果を表示
    print('回答:\n', result)


# main
if __name__ == '__main__':

    # Args
    args = parser.parse_args()

    if args.make_index:
        splitted_html = load_xmlfile(args.make_index)
        data = create_docs_from_html(splitted_html)
        create_store(data)
    
    elif args.query:
        query(args.query)

    elif args.retrieve:
        retrieve(args.retrieve)
