# gpt-3-especulação
Uma especulação de como seria o código fonte em Python do GPT-3


**Paridade**

Existem informações de que o GPT-3 tem 96 camadas, mas não temos a quantidade de "cabeças" em cada camada.
Assim, colocamos em nosso código 16 cabeças e 96 camadas.

• Usamos também a biblioteca Tensorflow, pois é da `Google`, que foi a que propôs e ajudou a desenvolver a arquitetura `Transformer` usada pelo GPT-3.

• Esse código não é 100% fiel e não está 100% completo, pois não tem o sistema de treinamento e ajuste de pesos e perdas.

**Inconformidades**

• seq_length = 2048. 2048 pode ser curto para o GPT-3. Essa função determina o quanto o modelo pode armazenar dados para lembrar do que ele e o usuário disseram.

• Tamanho de camadas ocultas (3072): Pode ser pouco para um modelo complexo como o GPT-3.

• Quantidade de cabeças (1536 | 96*16): Pode também ser pouco para um modelo grande como o GPT-3.

**Dataset**

Podemos especular que o GPT-3 foi treinado usando datasets como:

• Common Crawl: Um corpus massivo de centenas de bilhões de palavras coletadas da Web. O GPT-2 foi treinado no Common Crawl, então é muito provável que o GPT-3 também o tenha usado, possivelmente em uma versão maior e mais recente.

• Git public archives: Arquivos e commits marcados como públicos no Github (como este). Forneceria ensinamento sobre programação.

• Wikipedia: O corpus de textos da Wikipedia contém bilhões de palavras em uma variedade de tópicos.

• Project Gutenberg: Oferece mais de 60 mil obras literárias eletrônicas. Seria uma fonte rica em texto de alta qualidade para treinar o modelo de linguagem.

• Reddit Comments: Os comentários do Reddit contêm discussões informais organizadas por tópico. Útil para exposição à linguagem casual real.

• Twitter: Os tweets contêm uma enorme quantidade de linguagem informal curta. Contribuindo para o treinamento do modelo.

Muitos outros datasets podem ter sido usados, até mesmo dados privados. Sabemos que foram usados no GPT-3 **570 GB** de texto. O GPT-2 usou **40 GB**. Como base, a bíblia inteira, com aproximadamente **773.693** palavras e mais de **3.000.000** (três milhões) de letras/caracteres, tem de **4 a 5 MB**.


**C++ (cpp)**

Também podemos especular que o GPT-3 tenha sido escrito em `C++`. Pois é uma linguagem de baixo nivel que pode ajudar em otimização de `hardware`. 
Com uma linguagem de baixo nivel, pode ficar mais leve de ser execultado e trabalhar melhor com Tensores. Melhoraria também à portabilidade já que ele precisa de vários computadores diferentes para rodar o modelo. Podemos reforçar isso com o GPT do Bing. Já que foi feito uma portabilidade. Também modelos como o `LLama` da `Meta AI` que é escrito em `C++ e python`.

