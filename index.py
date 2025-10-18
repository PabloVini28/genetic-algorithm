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

ADAPTAÇÕES PARA TSPLIB95:
- Carregamento automático de arquivos .tsp
- Suporte a diferentes tipos de distância
- Experimentos automatizados com coleta de dados
"""

import random
import time
import csv
import os
from datetime import datetime

# Importação da biblioteca TSPLIB95 (será instalada se necessário) - *Recomendação do professor*
try:
    import tsplib95
    TSPLIB_DISPONIVEL = True
except ImportError:
    TSPLIB_DISPONIVEL = False
    print("Biblioteca tsplib95 não encontrada. Execute: pip install tsplib95")

class Grafo:
    """
    CLASSE GRAFO
    ============
    Representa o grafo do problema do caixeiro viajante.
    Contém as cidades e a matriz de distâncias entre elas.
    
    Atributos:
    - nome: Nome identificador do grafo
    - origem_tsplib: Indica se foi carregado da TSPLIB95
    """

    def __init__(self, nome, problema_tsplib=None):
        """
        Inicializa o grafo com nome, cidades e matriz de distâncias.
        Pode ser criado manualmente ou a partir de um problema TSPLIB95.
        
        Args:
            nome (str): Nome do grafo
            nomes_cidades (list): Lista de nomes das cidades (para grafos manuais)
            matriz_distancias (list): Matriz de distâncias entre cidades (para grafos manuais)
            problema_tsplib: Problema carregado da TSPLIB95
        """
        self.nome = nome
        self.origem_tsplib = problema_tsplib is not None
        
        if self.origem_tsplib:
            # Carregar de TSPLIB95 - carregar da pasta TSPLIB95
            self._carregar_de_tsplib(problema_tsplib)
    
    def _carregar_de_tsplib(self, problema):
        """
        Carrega dados de um problema TSPLIB95. // Basicamente eu crio a matriz de distâncias aqui
        """
        self.dimensao = problema.dimension
        self.tipo = problema.type
        self.edge_weight_type = getattr(problema, 'edge_weight_type', 'UNKNOWN')
        
        # Cria nomes das cidades baseados em índices
        self.nomes_cidades = [str(i+1) for i in range(self.dimensao)]
        
        # Calcula matriz de distâncias usando a função do tsplib95
        self.matriz_distancias = [[0] * self.dimensao for _ in range(self.dimensao)]

        graph = problema.get_graph()

        # Percorre as arestas e grava na matriz de adjacência
        for u, v, data in graph.edges(data=True):
            try:
                peso = data.get("weight", None)
                self.matriz_distancias[u-1][v-1] = peso
                self.matriz_distancias[v-1][u-1] = peso
            except Exception:
                print(f":: Erro na leitura da aresta ({u},{v})")


    def obter_distancia(self, cidade_origem_idx, cidade_destino_idx):
        """
        Retorna a distância entre duas cidades pelos seus índices.
        """
        return self.matriz_distancias[cidade_origem_idx][cidade_destino_idx]

    def obter_nome_cidade(self, cidade_idx):
        """
        Retorna o nome da cidade pelo seu índice.
        """
        return self.nomes_cidades[cidade_idx]

    def total_cidades(self):
        """
        Retorna o número total de cidades no grafo.
        """
        return len(self.nomes_cidades)

    def cromossomo_para_nomes(self, cromossomo):
        """
        Converte um cromossomo (lista de índices) para uma lista de nomes de cidades.
        """
        return [self.obter_nome_cidade(indice) for indice in cromossomo]
    
    @classmethod
    def carregar_tsplib(cls, caminho_arquivo):
        """
        Carrega um grafo de um arquivo TSPLIB95.
            
        Retorna um Grafo : Instância do grafo carregado
        """
        if not TSPLIB_DISPONIVEL:
            raise ImportError("Biblioteca tsplib95 não está disponível")
        
        try:
            problema = tsplib95.load(caminho_arquivo)
            nome = problema.name if problema.name else os.path.basename(caminho_arquivo).replace('.tsp', '') # Se não tiver nome, define o próprio nome do arquivo
            return cls(nome, problema_tsplib=problema)
        except Exception as e:
            raise Exception(f"Erro ao carregar {caminho_arquivo}: {e}")


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

    def __init__(self, grafo, geracao=0, cromossomo_previo=None):
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
        if cromossomo_previo is None:
            self.cromossomo = list(range(self.grafo.total_cidades()))
            random.shuffle(self.cromossomo)
        else:
            self.cromossomo = cromossomo_previo

    def calcular_fitness(self):
        """
        Calcula o fitness do indivíduo (distância total percorrida).
        Critério  Returns:
            float: Distância total percorrida na rotautilizado : menor distância = melhor fitness.
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
        filho1 = Individuo(self.grafo, self.geracao + 1, filho1_cromossomo)
        filho2 = Individuo(self.grafo, self.geracao + 1, filho2_cromossomo)

        return [filho1, filho2]

    def _preencher_genes_filho(self, cromossomo_filho, cromossomo_pai, tamanho):
        """
        Método auxiliar para preencher os genes restantes no crossover.
        Preenche as posições vazias (-1) com genes do pai, mantendo a ordem.
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

    def __init__(self, grafo, tamanho_populacao=50, stagnacao=250):
        """
        Inicializa o algoritmo genético.
        """
        self.grafo = grafo
        self.tamanho_populacao = tamanho_populacao
        self.populacao = []
        self.geracao = 0
        self.melhor_solucao = None
        self.stagnacao = stagnacao


    def _greedy_tour(self, dist):
        """
        Constrói um tour do TSP usando a heurística gulosa de adição de arestas.
        Entrada:
            dist: matriz NxN de distâncias (simétrica).
        Saída:
            tour: lista de vértices representando o ciclo Hamiltoniano.
        """
        n = len(dist)

        # --- 1. Cria lista de todas as arestas (i,j) com i < j, ordenadas por distância ---
        edge_list = [(dist[i][j], i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edge_list)
        edges = sorted(edge_list, key=lambda x: x[0])

        # --- 2. Inicializações ---
        degree = [0] * n              # grau de cada vértice
        parent = list(range(n))       # para união-busca (detecta ciclos)
        edges_in_tour = []            # arestas escolhidas (i,j)

        # Funções auxiliares de Union-Find ------------------------
        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u: int, v: int):
            ru, rv = find(u), find(v)
            parent[rv] = ru

        # --- 3. Percorre as arestas em ordem crescente ---
        for _, i, j in edges:
            if degree[i] == 2 or degree[j] == 2:
                continue  # já saturado
            ri, rj = find(i), find(j)

            # Verifica se adiciona aresta:
            # (a) Não cria ciclo prematuro (exceto na última adição)
            if ri == rj:
                # só podemos fechar o ciclo se for a última aresta (N-1 adicionadas)
                if len(edges_in_tour) == n - 1:
                    edges_in_tour.append((i, j))
                    degree[i] += 1
                    degree[j] += 1
                else:
                    continue
            else:
                # conecta os componentes
                union(i, j)
                edges_in_tour.append((i, j))
                degree[i] += 1
                degree[j] += 1

            # Parar quando tivermos N arestas → tour completo
            if len(edges_in_tour) == n:
                break

        # --- 4. Reconstruir o tour a partir das arestas escolhidas ---
        # Construir lista de adjacência
        adj = [[] for _ in range(n)]
        for i, j in edges_in_tour:
            adj[i].append(j)
            adj[j].append(i)

        # Encontrar o ciclo iniciando em 0
        tour = [0]
        prev, current = -1, 0
        while True:
            neighbors = adj[current]
            next_vertex = neighbors[0] if neighbors[0] != prev else neighbors[1]
            tour.append(next_vertex)
            if next_vertex == 0:
                break
            prev, current = current, next_vertex

        return tour[:-1]  # remove repetição do 0 final


    def _inicializar_populacao(self):
        """
        Cria a população inicial com indivíduos aleatórios.
        Cada indivíduo representa uma rota aleatória pelas cidades.
        """
        for _ in range(self.tamanho_populacao-3):
            self.populacao.append(Individuo(self.grafo))

        for _ in range(3):
            self.populacao.append(Individuo(self.grafo, cromossomo_previo=self._greedy_tour(self.grafo.matriz_distancias)))
        

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
        
        Retorna uma tupla com dois indivíduos selecionados como pais
        """
        # seleção por torneio
        tamanho_torneio = 5
        
        def realizar_torneio():
            """Realiza um torneio e retorna o vencedor."""
            competidores = random.sample(self.populacao, tamanho_torneio)
            competidores.sort(key=lambda ind: ind.distancia_percorrida)
            return competidores[0]  # retornTítuloa o melhor do torneio

        pai1 = realizar_torneio()
        pai2 = realizar_torneio()
        return pai1, pai2
    

    def _two_opt_first_improvement(self, tour, dist):
        """
        Heurística de busca local 2-OPT
        Parâmetros da entrada:
            tour - lista dos vértices (cromossomo)
            dist - matriz de distâncias
        """
        n = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n if i > 0 else n - 1):  # evita trocar o par (n-1,0)
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1) % n]
                    delta = dist[a][b] + dist[c][d] - (dist[a][c] + dist[b][d])
                    if delta > 1e-12:  # melhoria
                        # Reverter segmento tour[i+1 : j+1]
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def executar(self, num_geracoes, taxa_mutacao, verbose=True):
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
        4. Retorna métricas coletadas
            
        Retorna:
            Dicionário com métricas:
                - 'melhor_solucao': Valor da melhor solução encontrada
                - 'tempo_execucao': Tempo total de execução em segundos
                - 'rota': Sequência de cidades da melhor rota
        """
        inicio_tempo = time.time()
        
        if verbose:
            print(f"--- Resolvendo para o Grafo: {self.grafo.nome} ---")
        
        # PASSO 1: Geração da população inicial
        self._inicializar_populacao()

        # PASSO 2: Avaliação da população inicial
        for individuo in self.populacao:
            individuo.calcular_fitness()
        
        self._ordenar_populacao()
        self.melhor_solucao = self.populacao[0]
        
        if verbose:
            print(f"Geração 0 | Melhor distância: {self.melhor_solucao.distancia_percorrida}")

        contador_stagnacao = 0

        # PASSO 3: Loop evolutivo - executa por num_geracoes
        for geracao_atual in range(1, num_geracoes + 1):

            nova_populacao = []
            
            # ELITISMO: mantém os melhores indivíduos (10% da população)
            elite_size = int(self.tamanho_populacao * 0.1)
            nova_populacao.extend(self.populacao[:elite_size])

            # 2-OPT Local search
            # Realiza uma busca local nos 10% melhores da população 
            for individuo in nova_populacao:
                individuo.cromossomo[:] = self._two_opt_first_improvement(individuo.cromossomo, self.grafo.matriz_distancias)
                

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
                contador_stagnacao = 0
            else:
                contador_stagnacao += 1
            
            # RELATÓRIO: mostra progresso a cada 100 gerações
            if verbose and geracao_atual % 100 == 0:
                print(f"Geração {geracao_atual} | Melhor distância: {self.melhor_solucao.distancia_percorrida}")

            # Aborta geração se o algoritmo estiver estagnado
            if contador_stagnacao >= self.stagnacao:
                break

        # Calcula tempo total de execução
        tempo_execucao = time.time() - inicio_tempo
        
        # PASSO 4: Apresenta o resultado final (se verbose)
        if verbose:
            self.apresentar_resultado_final()
        
        # Retorna métricas coletadas
        return {
            'melhor_solucao': self.melhor_solucao.distancia_percorrida,
            'tempo_execucao': tempo_execucao,
            'rota': self.melhor_solucao.cromossomo.copy()
        }

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


# =================================================================
# FUNÇÕES PARA EXPERIMENTOS COM TSPLIB95
# =================================================================

def carregar_grafos_tsplib(diretorio="TSPLIB95", limite_vertices=400):
    """
    Carrega todos os grafos TSP válidos do diretório especificado.
    
    Retorna lista de objetos Grafo carregados da TSPLIB95
    """
    if not TSPLIB_DISPONIVEL:
        print("Biblioteca tsplib95 não disponível!")
        return []
    
    if not os.path.exists(diretorio):
        print(f"Diretório {diretorio} não encontrado!")
        return []
    
    grafos = []
    # Lista de arquivos removida - problema de formato foi corrigido
    
    arquivos_tsp = [f for f in os.listdir(diretorio) if f.endswith('.tsp')]
    arquivos_tsp.sort()
    
    print(f"Encontrados {len(arquivos_tsp)} arquivos TSP em {diretorio}")
    
    for arquivo in arquivos_tsp:
        caminho = os.path.join(diretorio, arquivo)
        
        try:
            grafo = Grafo.carregar_tsplib(caminho)
            
            if grafo.total_cidades() > limite_vertices:
                print(f"Ignorando {arquivo}: {grafo.total_cidades()} vértices > {limite_vertices}")
                continue
            
            grafos.append(grafo)
            print(f"Carregado: {arquivo} ({grafo.total_cidades()} cidades)")
            
        except Exception as e:
            print(f"Erro ao carregar {arquivo}: {str(e)[:50]}...")
    
    print(f"\nTotal de grafos carregados: {len(grafos)}")
    return grafos


def executar_experimentos_tsplib(grafos=None, num_execucoes=10, arquivo_csv="resultados_experimentos.csv"):
    """
    Executa experimentos automatizados com os grafos da TSPLIB95.
    
    MÉTRICAS COLETADAS:
    1. Valor da melhor solução calculada pelo GA
    2. Tempo de execução da melhor solução
    """
    
    # Carrega grafos automaticamente se não fornecidos
    if grafos is None:
        grafos = carregar_grafos_tsplib()
    
    if not grafos:
        print("Nenhum grafo disponível para experimentos!")
        return []
    
    # Parâmetros do algoritmo genético
    GERACOES = 500
    TAXA_MUTACAO = 0.02 # Padrão 
    TAMANHO_POPULACAO = 100
    MAX_STAGNACAO = 200
    
    print(f"\nINICIANDO EXPERIMENTOS AUTOMATIZADOS")
    print(f"Configuração:")
    print(f"   - Grafos: {len(grafos)}")
    print(f"   - Execuções por grafo: {num_execucoes}")
    print(f"   - Gerações: {GERACOES}")
    print(f"   - Taxa de mutação: {TAXA_MUTACAO*100}%")
    print(f"   - População: {TAMANHO_POPULACAO}")
    print(f"   - Arquivo CSV: {arquivo_csv}")
    print("=" * 60)
    
    resultados = []
    total_experimentos = len(grafos) * num_execucoes
    experimento_atual = 0
    
    for i, grafo in enumerate(grafos, 1):
        print(f"\n[{i}/{len(grafos)}] Processando: {grafo.nome}")
        print(f"Cidades: {grafo.total_cidades()}")
        print(f"Executando {num_execucoes} vezes...")
        
        for execucao in range(1, num_execucoes + 1):
            experimento_atual += 1
            
            print(f"Execução {execucao}/{num_execucoes} ", end="", flush=True)
            
            # Configura semente aleatória para reprodutibilidade
            random.seed(time.time() * 1000 + experimento_atual)
            
            # Cria e executa o algoritmo genético
            ag = AlgoritmoGenetico(grafo, tamanho_populacao=TAMANHO_POPULACAO, stagnacao=MAX_STAGNACAO)
            resultado = ag.executar(num_geracoes=GERACOES, taxa_mutacao=TAXA_MUTACAO, verbose=False)
            
            # Salva resultado com as métricas solicitadas
            registro = {
                'nome_grafo': grafo.nome,
                'num_cidades': grafo.total_cidades(),
                'execucao': execucao,
                'melhor_solucao': resultado['melhor_solucao'],  # MÉTRICA 1
                'tempo_execucao': round(resultado['tempo_execucao'], 4),  # MÉTRICA 2
                'data_execucao': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            resultados.append(registro)
            
            print(f"Distância: {resultado['melhor_solucao']} | Tempo: {resultado['tempo_execucao']:.2f}s")
        
        # Estatísticas do grafo atual
        resultados_grafo = [r for r in resultados if r['nome_grafo'] == grafo.nome]
        distancias = [r['melhor_solucao'] for r in resultados_grafo]
        tempos = [r['tempo_execucao'] for r in resultados_grafo]
        
        print(f"Melhor: {min(distancias)} | Pior: {max(distancias)} | Média: {sum(distancias)/len(distancias):.1f}")
        print(f"Tempo médio: {sum(tempos)/len(tempos):.2f}s")
    
    # Salva resultados em CSV
    print(f"\nSalvando resultados em {arquivo_csv}...")
    
    with open(arquivo_csv, 'w', newline='', encoding='utf-8') as arquivo:
        campos = ['nome_grafo', 'num_cidades', 'execucao', 'melhor_solucao', 'tempo_execucao', 'data_execucao']
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        
        escritor.writeheader()
        escritor.writerows(resultados)
    
    print(f"Arquivo CSV salvo com {len(resultados)} registros")
    print(f"Experimentos concluídos!")
    
    return resultados


if __name__ == "__main__":
    """
    EXECUÇÃO PRINCIPAL DO PROGRAMA
    ==============================
    Executar experimentos automatizados com TSPLIB95
    
    Para experimentos TSPLIB95, execute: python index.py --experimentos
    """
    
    import sys
    
    # Verifica se deve executar experimentos TSPLIB95
    if len(sys.argv) > 1 and sys.argv[1] == '--experimentos':
        
        print("EXPERIMENTOS AUTOMATIZADOS - TSPLIB95")
        print("=" * 60)
        print("Executando algoritmo genético 10 vezes para cada grafo")
        print("Coletando: melhor solução + tempo de execução")
        print("=" * 60)
        
        # Executa experimentos TSPLIB95
        resultados = executar_experimentos_tsplib(
            grafos=None,  # Carrega automaticamente
            num_execucoes=10,  # 10 execuções por grafo conforme solicitado
            arquivo_csv="resultados_experimentos_tsplib95.csv"
        )
        
        if resultados:
            print(f"\nRESUMO DOS EXPERIMENTOS:")
            print(f"Total de execuções: {len(resultados)}")
            print(f"Grafos testados: {len(set(r['nome_grafo'] for r in resultados))}")
            print(f"Arquivo gerado: resultados_experimentos_tsplib95.csv")
            
            # Estatísticas gerais
            print(f"\nESTATÍSTICAS GERAIS:")
            for grafo_nome in sorted(set(r['nome_grafo'] for r in resultados)):
                resultados_grafo = [r for r in resultados if r['nome_grafo'] == grafo_nome]
                distancias = [r['melhor_solucao'] for r in resultados_grafo]
                tempos = [r['tempo_execucao'] for r in resultados_grafo]
                
                print(f"   {grafo_nome:<12}: "
                      f"Melhor={min(distancias):6.0f} | "
                      f"Média={sum(distancias)/len(distancias):6.0f} | "
                      f"Tempo={sum(tempos)/len(tempos):.2f}s")