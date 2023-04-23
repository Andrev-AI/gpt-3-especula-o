# gpt-3-especulação
Uma especulação de como seria o código fonte em Python do GPT-3


**Paridade**

Existem informações que o GPT-3 tem 96 camadas, mas não temos o a quantidade de "cabeças" em cada camada. 
Assim colocamos em nosso código 16 cabeças e 96 camadas. 

• Usamos também a biblioteca Tensorflow pois é da `Google` que foi a que propós e ajudou a desenvolver a arquitetura `Transformer` que é usado pelo GPT-3.

• Esse código não é 100% fiel e não está 100% completo pois não tem o sistema de treinamento e ajute de pesos e perdas.

**Dataset**

Podemos especular que o GPT-3 foi treinado usando datasets como:

• Commun Crawl: Um corpus massivo de centenas de bilhões de palavras coletadas da Web. O GPT-2 foi treinado no Common Crawl, então é muito provável que o GPT-3 também o usasse, possivelmente uma versão maior e mais recente.

• Git public archives: Arquivos e commints marcados como públicos no github (como esse). Forneceria ensinamento sobre programação.

• Wikepédia: O corpus de textos da Wikipedia contém bilhões de palavras em uma variedade de tópicos.

• Project Gutenberg: Oferece mais de 60 mil obras literárias eletrônicas. Seria uma fonte rica em texto de alta qualidade para treinar o modelo de linguagem.

• Reddit Comments: Os comentários do Reddit contêm discussões informais organizadas por tópico. Útil para exposição à linguagem casual real.

• Twitter: Os tweets contêm uma enorme quantidade de linguagem informal curta. Contribuindo para o treinamento do modelo.

Muitos outros datasets poderiam ser usadas, dados até privados. Sabemos que foram usados no GPT-3 **570 GB** de texto. O GPT-2 usou **40 GB**. Como base, a bíblia inteira com aproximadamente 773.693 palavras e mais de 3.000.000 (três milhões) de letras/caracteres, tem de **4 a 5 MB**.



