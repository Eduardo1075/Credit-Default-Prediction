# Credit-Default-Prediction
![Predição de inadimplência](images/Gemini_Generated_Image_ygd2wnygd2wnygd2.png)

Este projeto consiste na análise de dados de uma empresa de cartão de crédito, que disponibilizou informações financeiras e comportamentais de 30 mil clientes ao longo de um período de 6 meses.
O principal objetivo deste estudo é o desenvolvimento de um modelo preditivo que seja capaz de antecipar a inadimplência de um cliente, ou seja, prever se determinada conta apresentará ou não atraso no pagamento no próximo mês.
Essa previsão é fundamental para que a empresa consiga mitigar riscos financeiros, ajustar suas políticas de concessão de crédito e adotar medidas preventivas, como redefinir limites de crédito, oferecer renegociação ou até mesmo negar novas compras a clientes com alto risco de inadimplência.
O modelo de machine learning que será desenvolvido terá como variável alvo a inadimplência no pagamento, uma variável binária que indica se o cliente ficou ou não inadimplente no período analisado. A partir disso, será possível gerar previsões com alto potencial de auxiliar a empresa na tomada de decisões estratégicas e na gestão de riscos.
Esse tipo de análise é muito utilizado no setor financeiro, especialmente por bancos e operadoras de crédito, para melhorar a eficiência operacional e garantir a sustentabilidade financeira da organização.

## 1. Descrição

Este é um projeto de machine learning de ponta a ponta que tem como objetivo prever a **probabilidade de inadimplência** de clientes de um serviço de cartão de crédito. O modelo de classificação é treinado com dados rotulados, onde o alvo é `1` caso o cliente se torne inadimplente no mês seguinte e `0` caso contrário.

A análise foi desenvolvida inicialmente em **notebooks Jupyter**, abordando todas as etapas do pipeline de ciência de dados: da **análise exploratória (EDA)** à **seleção de variáveis** e **modelagem preditiva**. Em seguida, o projeto foi estruturado de forma **modular**, com componentes separados para **ingestão de dados**, **transformação** e **treinamento do modelo**, permitindo organização e reprodutibilidade.

Com base nesses componentes, foram implementados scripts para **automação das etapas de treinamento e predição**, facilitando a reexecução e possíveis integrações futuras. Foram seguidas boas práticas como:

- Uso de ambientes virtuais para isolamento de dependências;
- Tratamento de exceções e geração de logs;
- Documentação clara de scripts, funções e classes;
- Estrutura de código limpa e bem organizada.

Embora **não tenha sido implementada uma API** neste projeto, sua estrutura modular permite fácil adaptação para esse tipo de integração, o que aproxima a solução de um **fluxo realista de trabalho em projetos de ciência de dados**.

---

## 2. Tecnologias e ferramentas utilizadas

As tecnologias e ferramentas utilizadas incluem:

- **Linguagem**: Python  
- **Bibliotecas**: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`  
- **Ambientes**: `Jupyter Notebook`, `Visual Studio Code`, `Anaconda`  
- **Controle de Versão**: `Git` e `GitHub`  

### Técnicas aplicadas:
- Classificação supervisionada  
- Estatística descritiva e teste ANOVA (F-test)  
- Avaliação com métricas como **precisão**, **recall**, **F1-score**  
- Análise da **distribuição de probabilidades previstas** e **curvas de threshold**

## 3. Problema de Negócio e Objetivo do Projeto

### 3.1 Qual é o problema de negócio?

Um banco está enfrentando dificuldades crescentes com clientes que não conseguem cumprir com os pagamentos mínimos de seus cartões de crédito, gerando inadimplência. A equipe de risco e crédito está em busca de soluções que ajudem a identificar, de forma antecipada, quais clientes têm maior probabilidade de se tornarem inadimplentes. Com essas informações, o banco poderá agir de forma proativa para mitigar riscos e reduzir prejuízos financeiros.

---

### 3.2 Qual é o contexto?

Quando um banco fornece crédito a um cliente, é fundamental monitorar e gerenciar indicadores de risco. Três indicadores-chave (KPIs) nesse processo são:

- **Taxa de Inadimplência (Default Rate):** Percentual de clientes que deixam de cumprir com suas obrigações financeiras em determinado período.
- **Custo de Crédito (Credit Cost):** Despesas que o banco incorre ao conceder crédito, incluindo perdas com inadimplentes e provisões para devedores duvidosos.
- **Retorno Ajustado ao Risco (RAROC):** Métrica que relaciona o retorno obtido sobre o capital ajustado ao risco da operação.

Para maximizar sua rentabilidade, o banco deve **reduzir a inadimplência**, **minimizar o custo de crédito** e **aumentar a eficiência das concessões**. Modelos preditivos são aliados estratégicos nesse desafio, permitindo maior precisão na concessão e monitoramento de crédito.

---

### 3.3 Quais são os objetivos do projeto?

- Identificar os principais fatores associados à inadimplência.
- Construir um modelo capaz de prever a probabilidade de um cliente se tornar inadimplente no mês seguinte.
- Fornecer insights e estratégias para ajudar o banco a reduzir sua taxa de inadimplência.
- Visualizar e interpretar os resultados do modelo com base em métricas e gráficos de apoio.

---

### 3.4 Quais são os benefícios do projeto?

- **Redução de prejuízos financeiros** relacionados a não pagamento de dívidas.
- **Melhoria na concessão de crédito**, com decisões mais informadas.
- **Ações preventivas** mais direcionadas para clientes em risco.
- **Aumento da eficiência operacional** do setor de análise de crédito.
- **Proteção da receita** e maior sustentabilidade financeira.

---

### 3.5 Conclusão

Ao implantar o modelo em ambiente produtivo, o objetivo principal é gerar **escores de probabilidade de inadimplência para cada cliente**. Essa abordagem é mais valiosa do que classificações binárias (inadimplente/não inadimplente), pois permite uma **gestão mais estratégica do risco**.

Por exemplo, com base na probabilidade prevista, o banco pode:

- Priorizar campanhas de renegociação para clientes com alto risco;
- Ajustar limites de crédito;
- Aplicar políticas de retenção personalizadas.

Dessa forma, a instituição financeira toma decisões mais inteligentes e **baseadas em dados**, aumentando a rentabilidade e a segurança de suas operações.

## 4. Pipeline da Solução

O pipeline adotado neste projeto segue a metodologia **CRISP-DM (Cross Industry Standard Process for Data Mining)**, uma abordagem estruturada amplamente utilizada em projetos de ciência de dados. As etapas aplicadas foram:

1. **Definição do problema de negócio**  
   Identificar clientes com alta probabilidade de inadimplência e fornecer subsídios para que o banco atue preventivamente.

2. **Coleta e compreensão dos dados**  
   Utilização de um conjunto de dados real com informações financeiras e comportamentais de 30 mil clientes, contendo atributos como limite de crédito, histórico de pagamento, valores de faturas e pagamentos realizados.

3. **Divisão em conjunto de treino e teste**  
   Separação dos dados para garantir avaliação imparcial do modelo e evitar overfitting.

4. **Análise exploratória dos dados (EDA)**  
   Investigação de correlações, identificação de outliers, comportamento das variáveis em relação à inadimplência e visualizações estatísticas.

5. **Engenharia de atributos e pré-processamento**  
   Inclusão de variáveis derivadas, limpeza de dados, normalização e transformação para otimizar a performance dos algoritmos.

6. **Treinamento, seleção e avaliação de modelos**  
   Aplicação de modelos de classificação supervisionada, uso de teste ANOVA para seleção de variáveis relevantes, comparação de métricas e ajustes de thresholds.

7. **Teste e validação do modelo final**  
   Avaliação com base em métricas como precisão, recall, F1-score e análise da distribuição das probabilidades previstas.

8. **Conclusão e interpretação dos resultados**  
   Interpretação dos padrões encontrados, identificação dos fatores mais relevantes e proposta de estratégias de mitigação de risco.

> Todas as etapas serão explicadas detalhadamente dentro do readme.md e nos notebooks, com o racional por trás das decisões tomadas em cada fase do desenvolvimento.

## 5. Análise das variáveis idade e limite de crédito
![histograma](images/transferir(19).png)
## 📊 Histograma da variável `AGE` (Idade)

### ✅ O que mostra:
- A maioria dos clientes está na faixa entre **25 e 40 anos**.
- A frequência **diminui gradualmente** após os 40.
- Há **pouquíssimos clientes acima dos 60 anos**.

### 📌 Conclusão:
A base de dados é predominantemente composta por **adultos jovens e de meia-idade**. Pode ser interessante verificar:
- Se a **inadimplência muda com a idade**.
- Se essa variável tem **correlação com o comportamento de crédito**.

---

## 📊 Histograma da variável `LIMIT_BAL` (Limite de crédito)

### ✅ O que mostra:
- A maioria dos clientes tem limite de até **200.000**.
- O pico ocorre abaixo de **100.000**, e vai **caindo à medida que o limite aumenta**.
- Limites muito altos (**> 500.000**) são **raros**.

### 📌 Conclusão:
A distribuição é **assimétrica à direita** (muito concentrada em valores baixos). Isso indica que:
- **Limites baixos são muito comuns**.
- Poucos clientes têm acesso a **grandes linhas de crédito**.
