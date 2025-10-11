"""
PROBLEMA DO CAIXEIRO VIAJANTE - ALGORITMO GENÉTICO
==================================================

Este código implementa uma solução para o Problema do Caixeiro Viajante (TSP - Traveling Salesman Problem)
usando Algoritmo Genético. O objetivo é encontrar a menor rota que visite todas as cidades exatamente
uma vez e retorne à cidade de origem.

COMPONENTES PRINCIPAIS:
1. Grafo: Representa as cidades e distâncias entre elas
2. Individuo: Representa uma possível solução (rota)
3. AlgoritmoGenetico: Implementa a evolução da população de soluções

TÉCNICAS UTILIZADAS:
- Crossover
- Mutação por troca de posições
- Seleção por torneio
- Elitismo (manutenção dos melhores indivíduos)
"""

import random
import json

class Grafo:
    """
    CLASSE GRAFO
    ============
    Representa o grafo do problema do caixeiro viajante.
    Contém as cidades e a matriz de distâncias entre elas.
    
    Atributos:
    - nome: Nome identificador do grafo
    - nomes_cidades: Lista com os nomes das cidades
    - matriz_distancias: Matriz quadrada com distâncias entre cidades
    """

    def __init__(self, nome, nomes_cidades, matriz_distancias):
        """
        Inicializa o grafo com nome, cidades e matriz de distâncias.
        
        Args:
            nome (str): Nome do grafo
            nomes_cidades (list): Lista de nomes das cidades
            matriz_distancias (list): Matriz de distâncias entre cidades
        """
        self.nome = nome
        self.nomes_cidades = nomes_cidades
        self.matriz_distancias = matriz_distancias

    def obter_distancia(self, cidade_origem_idx, cidade_destino_idx):
        """
        Retorna a distância entre duas cidades pelos seus índices.
        
        Args:
            cidade_origem_idx (int): Índice da cidade de origem
            cidade_destino_idx (int): Índice da cidade de destino
            
        Returns:
            float: Distância entre as cidades
        """
        return self.matriz_distancias[cidade_origem_idx][cidade_destino_idx]

    def obter_nome_cidade(self, cidade_idx):
        """
        Retorna o nome da cidade pelo seu índice.
        
        Args:
            cidade_idx (int): Índice da cidade
            
        Returns:
            str: Nome da cidade
        """
        return self.nomes_cidades[cidade_idx]

    def total_cidades(self):
        """
        Retorna o número total de cidades no grafo.
        
        Returns:
            int: Quantidade de cidades
        """
        return len(self.nomes_cidades)

    def cromossomo_para_nomes(self, cromossomo):
        """
        Converte um cromossomo (lista de índices) para uma lista de nomes de cidades.
        
        Args:
            cromossomo (list): Lista de índices das cidades
            
        Returns:
            list: Lista com nomes das cidades correspondentes
        """
        return [self.obter_nome_cidade(indice) for indice in cromossomo]


class Individuo:
    """
    CLASSE INDIVIDUO
    ================
    Representa um indivíduo que no caso seria uma solução candidata na  população do algoritmo genético.
    Cada indivíduo possui um cromossomo que representa uma rota através das cidades.
    
    Atributos:
    - grafo: Referência ao grafo do problema
    - geracao: Número da geração em que o indivíduo foi criado
    - distancia_percorrida: Distância total da rota (fitness)
    - cromossomo: Lista de índices representando a ordem das cidades a visitar
    """

    def __init__(self, grafo, geracao=0):
        """
        Inicializa um indivíduo com uma rota aleatória.
        
        Args:
            grafo (Grafo): Grafo do problema
            geracao (int): Número da geração (padrão: 0)
        """
        self.grafo = grafo
        self.geracao = geracao
        self.distancia_percorrida = 0
        
        # cria um cromossomo com uma rota aleatória sem repetição de cidades
        # exemplo: [0, 3, 1, 2] significa visitar cidade 0 → 3 → 1 → 2 → 0 na ordem que está no vetor
        self.cromossomo = list(range(self.grafo.total_cidades()))
        random.shuffle(self.cromossomo)

    def calcular_fitness(self):
        """
        Calcula o fitness do indivíduo (distância total percorrida).
        Critério utilizado : menor distância = melhor fitness.
        
        Returns:
            float: Distância total percorrida na rota
        """
        distancia_total = 0
        for i in range(len(self.cromossomo)):
            cidade_origem = self.cromossomo[i]
            # Se é a última cidade, volta para a primeira
            if i == len(self.cromossomo) - 1:
                cidade_destino = self.cromossomo[0]
            else:
                cidade_destino = self.cromossomo[i + 1]
            
            distancia_total += self.grafo.obter_distancia(cidade_origem, cidade_destino)
            
        self.distancia_percorrida = distancia_total
        return self.distancia_percorrida

    def crossover(self, outro_individuo):
        """
        Realiza crossover com outro indivíduo.
        Gera dois filhos preservando a ordem relativa dos genes.
        
        Args:
            outro_individuo (Individuo): Outro pai para o crossover
            
        Returns:
            list: Lista com dois filhos resultantes do crossover
        """
        tamanho = len(self.cromossomo)
        filho1_cromossomo = [-1] * tamanho
        filho2_cromossomo = [-1] * tamanho

        # seleciona dois pontos de corte aleatórios
        ponto1, ponto2 = sorted(random.sample(range(tamanho), 2))

        # copia o segmento entre os pontos de corte dos pais para os filhos
        filho1_cromossomo[ponto1:ponto2] = self.cromossomo[ponto1:ponto2]
        filho2_cromossomo[ponto1:ponto2] = outro_individuo.cromossomo[ponto1:ponto2]

        # preenche o restante dos genes mantendo a ordem
        self._preencher_genes_filho(filho1_cromossomo, outro_individuo.cromossomo, tamanho)
        self._preencher_genes_filho(filho2_cromossomo, self.cromossomo, tamanho)
        
        # cria os filhos
        filho1 = Individuo(self.grafo, self.geracao + 1)
        filho2 = Individuo(self.grafo, self.geracao + 1)
        filho1.cromossomo = filho1_cromossomo
        filho2.cromossomo = filho2_cromossomo

        return [filho1, filho2]

    def _preencher_genes_filho(self, cromossomo_filho, cromossomo_pai, tamanho):
        """
        Método auxiliar para preencher os genes restantes no crossover.
        Preenche as posições vazias (-1) com genes do pai, mantendo a ordem.
        
        Args:
            cromossomo_filho (list): Cromossomo do filho a ser preenchido
            cromossomo_pai (list): Cromossomo do pai para buscar genes
            tamanho (int): Tamanho do cromossomo
        """
        ponteiro_pai = 0
        for i in range(tamanho):
            if cromossomo_filho[i] == -1:
                # encontra o próximo gene do pai que ainda não está no filho
                while cromossomo_pai[ponteiro_pai] in cromossomo_filho:
                    ponteiro_pai += 1
                cromossomo_filho[i] = cromossomo_pai[ponteiro_pai]

    def mutar(self, taxa_mutacao):
        """
        Aplica mutação por troca de posições.
        Com uma probabilidade definida pela taxa de mutação, troca duas cidades de posição.
        
        Args:
            taxa_mutacao (float): Probabilidade de ocorrer mutação (0.0 a 1.0)
        """
        if random.random() < taxa_mutacao:
            # seleciona duas posições aleatórias e troca as cidades
            idx1, idx2 = random.sample(range(len(self.cromossomo)), 2)
            self.cromossomo[idx1], self.cromossomo[idx2] = self.cromossomo[idx2], self.cromossomo[idx1]


class AlgoritmoGenetico:
    """
    CLASSE ALGORITMO GENÉTICO
    =========================
    Implementa o algoritmo genético para resolver o problema do caixeiro viajante.
    
    FUNCIONAMENTO:
    1. Inicializa uma população aleatória de indivíduos
    2. Avalia o fitness de cada indivíduo
    3. Para cada geração:
       - Seleciona pais por torneio
       - Aplica crossover para gerar filhos
       - Aplica mutação nos filhos
       - Aplica elitismo (mantém os melhores)
       - Substitui a população antiga pela nova
    4. Repete até atingir o número máximo de gerações
    
    Atributos:
    - grafo: Grafo do problema
    - tamanho_populacao: Número de indivíduos na população
    - populacao: Lista de indivíduos atuais
    - geracao: Número da geração atual
    - melhor_solucao: Melhor indivíduo encontrado até o momento
    """

    def __init__(self, grafo, tamanho_populacao=50):
        """
        Inicializa o algoritmo genético.
        
        Args:
            grafo (Grafo): Grafo do problema a ser resolvido
            tamanho_populacao (int): Tamanho da população (padrão: 50)
        """
        self.grafo = grafo
        self.tamanho_populacao = tamanho_populacao
        self.populacao = []
        self.geracao = 0
        self.melhor_solucao = None

    def _inicializar_populacao(self):
        """
        Cria a população inicial com indivíduos aleatórios.
        Cada indivíduo representa uma rota aleatória pelas cidades.
        """
        for _ in range(self.tamanho_populacao):
            self.populacao.append(Individuo(self.grafo))

    def _ordenar_populacao(self):
        """
        Ordena a população por fitness (distância percorrida).
        Indivíduos com menor distância ficam no início da lista.
        """
        self.populacao.sort(key=lambda ind: ind.distancia_percorrida)

    def _selecionar_pais(self):
        """
        Seleciona dois pais usando seleção por torneio.
        
        SELEÇÃO POR TORNEIO:
        - Seleciona aleatoriamente 5 indivíduos
        - Escolhe o melhor entre eles
        - Repete para selecionar o segundo pai
        
        Returns:
            tuple: Tupla com dois indivíduos selecionados como pais
        """
        # seleção por torneio
        tamanho_torneio = 5
        
        def realizar_torneio():
            """Realiza um torneio e retorna o vencedor."""
            competidores = random.sample(self.populacao, tamanho_torneio)
            competidores.sort(key=lambda ind: ind.distancia_percorrida)
            return competidores[0]  # retorna o melhor do torneio

        pai1 = realizar_torneio()
        pai2 = realizar_torneio()
        return pai1, pai2

    def executar(self, num_geracoes, taxa_mutacao):
        """
        Executa o algoritmo genético por um número especificado de gerações.
        
        PIPELINE DE EXECUÇÃO:
        1. Inicialização da população
        2. Avaliação inicial
        3. Loop evolutivo:
           - Seleção de pais
           - Crossover
           - Mutação
           - Elitismo
           - Substituição populacional
           - Avaliação da nova geração
        4. Apresentação do resultado final
        
        Args:
            num_geracoes (int): Número de gerações a evoluir
            taxa_mutacao (float): Taxa de mutação (0.0 a 1.0)
        """
        print(f"--- Resolvendo para o Grafo: {self.grafo.nome} ---")
        
        # PASSO 1: Geração da população inicial
        self._inicializar_populacao()

        # PASSO 2: Avaliação da população inicial
        for individuo in self.populacao:
            individuo.calcular_fitness()
        
        self._ordenar_populacao()
        self.melhor_solucao = self.populacao[0]
        
        print(f"Geração 0 | Melhor distância: {self.melhor_solucao.distancia_percorrida}")

        # PASSO 3: Loop evolutivo - executa por num_geracoes
        for geracao_atual in range(1, num_geracoes + 1):
            nova_populacao = []
            
            # ELITISMO: mantém os melhores indivíduos (10% da população)
            elite_size = int(self.tamanho_populacao * 0.1)
            nova_populacao.extend(self.populacao[:elite_size])

            # REPRODUÇÃO: gera o restante da população via crossover e mutação
            while len(nova_populacao) < self.tamanho_populacao:
                # Seleção de pais
                pai1, pai2 = self._selecionar_pais()
                
                # Crossover - gera dois filhos
                filho1, filho2 = pai1.crossover(pai2)
                
                # Mutação nos filhos
                filho1.mutar(taxa_mutacao)
                filho2.mutar(taxa_mutacao)

                # Adiciona filhos à nova população
                nova_populacao.append(filho1)
                # garante que não ultrapassemos o tamanho da população
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(filho2)
            
            # SUBSTITUIÇÃO: nova população substitui a antiga
            self.populacao = nova_populacao

            # AVALIAÇÃO: calcula fitness da nova população
            for individuo in self.populacao:
                individuo.calcular_fitness()

            self._ordenar_populacao()

            # ATUALIZAÇÃO: atualiza a melhor solução se encontrou algo melhor
            if self.populacao[0].distancia_percorrida < self.melhor_solucao.distancia_percorrida:
                self.melhor_solucao = self.populacao[0]
            
            # RELATÓRIO: mostra progresso a cada 100 gerações
            if geracao_atual % 100 == 0:
                print(f"Geração {geracao_atual} | Melhor distância: {self.melhor_solucao.distancia_percorrida}")

        # PASSO 4: Apresenta o resultado final para o grafo atual
        self.apresentar_resultado_final()

    def apresentar_resultado_final(self):
        """
        Imprime a melhor solução encontrada com detalhes da rota.
        Mostra a distância total e a sequência de cidades a visitar.
        """
        print("\n--- Melhor solução encontrada ---")
        print(f"Distância total: {self.melhor_solucao.distancia_percorrida}")
        rota_nomes = self.grafo.cromossomo_para_nomes(self.melhor_solucao.cromossomo)
        print(f"Rota: {' -> '.join(rota_nomes)} -> {rota_nomes[0]}")
        print("-" * 35 + "\n")


if __name__ == "__main__":
    """
    EXECUÇÃO PRINCIPAL DO PROGRAMA
    ==============================
    
    Este bloco define os grafos de teste e executa o algoritmo genético para cada um.
    
    PIPELINE DE EXECUÇÃO COMPLETO:
    
    1. DEFINIÇÃO DOS GRAFOS
       - Cria diferentes instâncias do problema com cidades e distâncias
       - Cada grafo representa um cenário de teste diferente
    
    2. CONFIGURAÇÃO DOS PARÂMETROS
       - Define número de gerações e taxa de mutação
       - Estes parâmetros controlam o comportamento do algoritmo
    
    3. EXECUÇÃO PARA CADA GRAFO
       - Itera sobre todos os grafos definidos
       - Para cada grafo, executa o algoritmo genético
       - Apresenta os resultados de cada execução
    """
    
    # =================================================================
    # PASSO 1: DEFINIÇÃO DOS GRAFOS DE TESTE
    # =================================================================
    
    # Grafo 1: Exemplo pequeno com 5 cidades (A, B, C, D, E)
    # Matriz simétrica onde matriz[i][j] = distância da cidade i para cidade j
    grafo_pequeno = Grafo(
        nome="Grafo de Teste 5 Cidades",
        nomes_cidades=["A", "B", "C", "D", "E"],
        matriz_distancias=[
            #    A   B   C   D   E
            [0, 10, 15, 20, 25],  # Distâncias de A
            [10, 0, 35, 25, 30],  # Distâncias de B
            [15, 35, 0, 30, 10],  # Distâncias de C
            [20, 25, 30, 0, 5],   # Distâncias de D
            [25, 30, 10, 5, 0]    # Distâncias de E
        ]
    )

    # Grafo 2: Exemplo com cidades do Ceará
    # Baseado em distâncias aproximadas entre cidades reais
    grafo_simples = Grafo(
        nome="Grafo Simples 4 Cidades",
        nomes_cidades=["Fortaleza", "Quixadá", "Canindé", "Sobral"],
        matriz_distancias=[
            #      Fort  Quix  Cani  Sobr
            [0, 160, 120, 230], # Distâncias de Fortaleza
            [160, 0, 70, 250],  # Distâncias de Quixadá
            [120, 70, 0, 180],  # Distâncias de Canindé
            [230, 250, 180, 0]  # Distâncias de Sobral
        ]
    )
    
    # Grafo 3: Exemplo mais complexo com 6 pontos
    # Demonstra a capacidade do algoritmo em problemas maiores
    grafo_alternativo = Grafo(
        nome="Grafo Alternativo 6 Cidades",
        nomes_cidades=["P1", "P2", "P3", "P4", "P5", "P6"],
        matriz_distancias=[
            #    P1  P2  P3  P4  P5  P6
            [0, 29, 82, 46, 68, 52],  # Distâncias de P1
            [29, 0, 55, 46, 42, 42],  # Distâncias de P2
            [82, 55, 0, 68, 46, 29],  # Distâncias de P3
            [46, 46, 68, 0, 24, 29],  # Distâncias de P4
            [68, 42, 46, 24, 0, 24],  # Distâncias de P5
            [52, 42, 29, 29, 24, 0]   # Distâncias de P6
        ]
    )

    # =================================================================
    # PASSO 2: CONFIGURAÇÃO DOS PARÂMETROS DO ALGORITMO
    # =================================================================

    # Lista contendo todos os grafos que queremos resolver
    lista_de_grafos = [grafo_pequeno, grafo_simples, grafo_alternativo]
    
    # Parâmetros do Algoritmo Genético
    NUMERO_DE_GERACOES = 500    # Quantas gerações o algoritmo vai evoluir
    TAXA_DE_MUTACAO = 0.02      # 2% de chance de mutação para cada indivíduo
    TAMANHO_POPULACAO = 100     # Quantos indivíduos em cada geração

    # =================================================================
    # PASSO 3: EXECUÇÃO DO ALGORITMO PARA CADA GRAFO
    # =================================================================
    
    print("="*60)
    print("INICIANDO RESOLUÇÃO DO PROBLEMA DO CAIXEIRO VIAJANTE")
    print("USANDO ALGORITMO GENÉTICO")
    print("="*60)
    print(f"Parâmetros:")
    print(f"- Gerações: {NUMERO_DE_GERACOES}")
    print(f"- Taxa de Mutação: {TAXA_DE_MUTACAO*100}%")
    print(f"- Tamanho da População: {TAMANHO_POPULACAO}")
    print(f"- Grafos a resolver: {len(lista_de_grafos)}")
    print("="*60)

    # Itera sobre a lista de grafos e resolve cada um
    for i, grafo_para_resolver in enumerate(lista_de_grafos, 1):
        print(f"\n[{i}/{len(lista_de_grafos)}] Processando: {grafo_para_resolver.nome}")
        print(f"Número de cidades: {grafo_para_resolver.total_cidades()}")
        
        # Cria uma instância do algoritmo genético para este grafo
        ag = AlgoritmoGenetico(grafo=grafo_para_resolver, tamanho_populacao=TAMANHO_POPULACAO)
        
        # Executa o algoritmo
        ag.executar(num_geracoes=NUMERO_DE_GERACOES, taxa_mutacao=TAXA_DE_MUTACAO)
    
    print("="*60)
    print("EXECUÇÃO CONCLUÍDA!")
    print("Todas as instâncias do problema foram resolvidas.")
    print("="*60)