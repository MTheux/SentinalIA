import streamlit as st
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import requests
import time
import json
from urllib.parse import urlparse
import streamlit.components.v1 as components
import yaml
# import csv # Removido, pois não usaremos arquivos CSV locais


# --- Configurações do LLM (Temperatura Reduzida para Consistência) ---
LLM_TEMPERATURE = 0.1 

# --- Configuração do LLM (API Key) ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
NVD_API_KEY = os.getenv("NVD_API_KEY") # Carrega a NVD API Key
ACUNETIX_API_KEY = os.getenv("ACUNETIX_API_KEY") # Adicionado para carregar a Acunetix API Key
ACUNETIX_URL = os.getenv("ACUNETIX_URL") # Adicionado para carregar a URL do Acunetix

if not API_KEY:
    st.error("ERRO: A variável de ambiente 'GOOGLE_API_KEY' não está configurada.")
    st.info("Por favor, crie um arquivo .env na raiz do seu projeto e adicione 'GOOGLE_API_KEY=SUA_CHAVE_AQUI'.")
    st.info("Você pode obter sua chave em https://aistudio.google.com/app/apikey")
    st.stop()

# --- Dicionários de Referência da OWASP ---
OWASP_TOP_10_2021 = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable and Outdated Components",
    "A07": "Identification and Authentication Failures",
    "A08": "Software and Data Integrity Failures",
    "A09": "Security Logging and Monitoring Failures",
    "A10": "Server-Side Request Forgery (SSRF)"
}

OWASP_API_TOP_10_2023 = {
    "API1": "Broken Object Level Authorization (BOLA)",
    "API2": "Broken Authentication",
    "API3": "Broken Object Property Level Authorization",
    "API4": "Unrestricted Resource Consumption",
    "API5": "Broken Function Level Authorization (BFLA)",
    "API6": "Unrestricted Access to Sensitive Business Flows",
    "API7": "Server Side Request Forgery (SSRF)",
    "API8": "Security Misconfiguration",
    "API9": "Improper Inventory Management",
    "API10": "Unsafe Consumption of APIs"
}

OWASP_SUBCATEGORIES = {
    "A01": [
        "Insecure Direct Object References (IDOR)", "Missing Function Level Access Control",
        "Privilege Escalation (Vertical/Horizontal)", "Path Traversal",
        "URL Tampering", "Parameter Tampering"
    ],
    "A02": [
        "Weak Hashing Algorithms", "Use of Outdated/Weak Encryption Protocols (e.g., TLS 1.0/1.1)",
        "Hardcoded Cryptographic Keys", "Improper Key Management",
        "Exposure of Sensitive Data in Transit/At Rest"
    ],
    "A03": [
        "SQL Injection (SQLi)", "Cross-Site Scripting (XSS)",
        "Command Injection", "LDAP Injection", "XPath Injection",
        "NoSQL Injection", "Server-Side Template Injection (SSTI)",
        "Code Injection (e.g., PHP, Python, Java)", "Header Injection (e.g., Host Header Injection)"
    ],
    "A04": [
        "Business Logic Flaws", "Lack of Security Design Principles",
        "Trust Boundary Violations", "Feature Overload",
        "Insecure Direct Object References (IDOR) - (also A01, design aspect)"
    ],
    "A05": [
        "Default Passwords/Configurations", "Unnecessary Features/Services Enabled",
        "Improper File/Directory Permissions", "Missing Security Headers",
        "Error Messages Revealing Sensitive Information", "Open Cloud Storage Buckets"
    ],
    "A06": [
        "Using Libraries/Frameworks with Known Vulnerabilities", "Outdated Server Software (e.g., Apache, Nginx, IIS)",
        "Client-Side Libraries with Vulnerabilities", "Lack of Patch Management"
    ],
    "A07": [
        "Weak Password Policies", "Missing Multi-Factor Authentication (MFA)",
        "Session Management Flaws (e.g., fixed session IDs)", "Improper Credential Recovery Mechanisms",
        "Brute-Force Attacks (lack of rate limiting)"
    ],
    "A08": [
        "Insecure Deserialization", "Lack of Integrity Checks on Updates/Packages",
        "Weak Digital Signatures", "Client-Side Trust (e.g., relying on client-side validation)"
    ],
    "A09": [
        "Insufficient Logging of Security Events", "Lack of Alerting on Suspicious Activities",
        "Inadequate Retention of Logs", "Logs Not Protected from Tampering"
    ],
    "A10": "Server-Side Request Forgery (SSRF)"
}


# --- Funções Auxiliares Comuns ---

def get_gemini_models():
    if 'llm_models' not in st.session_state:
        st.session_state.llm_models = {'vision_model': None, 'text_model': None}

    if not st.session_state.llm_models_initialized:
        st.write("Configurando modelos Gemini...")
        genai.configure(api_key=API_KEY)

        llm_model_vision_temp = None
        llm_model_text_temp = None

        vision_model_priority = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]
        text_model_priority = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

        try:
            available_models = list(genai.list_models())
            
            for preferred_name in vision_model_priority:
                for m in available_models:
                    if preferred_name in m.name and 'generateContent' in m.supported_generation_methods:
                        llm_model_vision_temp = genai.GenerativeModel(m.name)
                        st.success(f"Modelo LLM para Visão selecionado: {m.name}")
                        break
                if llm_model_vision_temp:
                    break
            
            for preferred_name in text_model_priority:
                for m in available_models:
                    if preferred_name in m.name and 'generateContent' in m.supported_generation_methods:
                        llm_model_text_temp = genai.GenerativeModel(m.name, generation_config={"temperature": LLM_TEMPERATURE})
                        st.success(f"Modelo LLM para Texto selecionado: {m.name}")
                        break
                if llm_model_text_temp:
                    break

            if not llm_model_vision_temp:
                st.error("ERRO: Nenhum modelo LLM de visão adequado (gemini-1.5-flash/pro ou gemini-pro-vision) encontrado.")
                st.info("Verifique a disponibilidade de modelos no Google AI Studio.")
                st.stop()
            if not llm_model_text_temp:
                st.error("ERRO: Nenhum modelo LLM de texto adequado (gemini-1.5-flash/pro ou gemini-pro) encontrado.")
                st.info("Verifique a disponibilidade de modelos no Google AI Studio.")
                st.stop()

        except Exception as e:
            st.error(f"ERRO ao listar ou selecionar modelos do Gemini: {e}")
            st.info("Verifique sua conexão com a internet e sua chave de API.")
            st.stop()
        
        st.session_state.llm_models['vision_model'] = llm_model_vision_temp
        st.session_state.llm_models['text_model'] = llm_model_text_temp
        st.session_state.llm_models_initialized = True
    
    return st.session_state.llm_models['vision_model'], st.session_state.llm_models['text_model']


def obter_resposta_llm(model_instance, prompt_parts):
    if model_instance is None:
        st.error("Erro: O modelo LLM não foi inicializado corretamente. Não é possível gerar conteúdo.")
        return None
    try:
        response = model_instance.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Erro ao comunicar com o LLM: {e}")
        st.info("Verifique se a sua conexão com a internet está ativa e se o modelo LLM está funcionando.")
        return None

def formatar_resposta_llm(resposta_bruta):
    return resposta_bruta

@st.cache_data(show_spinner=False)
def mapear_falha_para_owasp(_llm_text_model, falha_input):
    owasp_list = "\n".join([f"{code}: {name}" for code, name in OWASP_TOP_10_2021.items()])
    
    prompt = (
        f"Qual categoria da OWASP Top 10 (2021) melhor representa a vulnerabilidade ou técnica de ataque '{falha_input}'?"
        f"\n\nConsidere a seguinte lista de categorias OWASP Top 10 (2021):"
        f"\n{owasp_list}"
        f"\n\nResponda apenas com o CÓDIGO da categoria OWASP (ex: A03) e nada mais. Se não tiver certeza ou se não se encaixar em nenhuma categoria, responda 'INDEFINIDO'."
        f"Exemplos: 'SQL Injection' -> 'A03', 'Cross-Site Scripting' -> 'A03', 'IDOR' -> 'A01', 'Clickjacking' -> 'A04'"
    )
    
    with st.spinner(f"Tentando mapear '{falha_input}' para uma categoria OWASP..."):
        resposta = obter_resposta_llm(_llm_text_model, [prompt])
    
    if resposta:
        codigo_owasp = resposta.strip().upper().split(':')[0].split(' ')[0]
        if codigo_owasp in OWASP_TOP_10_2021:
            return codigo_owasp
        elif codigo_owasp == "INDEFINIDO":
            st.warning("O LLM não conseguiu mapear a falha para uma categoria OWASP específica.")
            return None
        else:
            st.warning(f"O LLM retornou um código inesperado: '{codigo_owasp}'.")
            return None
    return None

def parse_vulnerability_summary(text_response):
    """
    Tenta extrair o resumo de vulnerabilidades (total e por criticidade) de uma resposta de texto.
    Espera um formato como: "Total de Vulnerabilidades: X | Críticas: Y | Altas: Z | Médias: W | Baixas: V"
    Retorna o dicionário de resumo e o texto da resposta SEM A LINHA DO RESUMO.
    """
    summary = {
        "Total": 0, "Críticas": 0, "Altas": 0, "Médias": 0, "Baixas": 0
    }
    
    lines = text_response.split('\n')
    summary_line_found = False
    parsed_content = []

    for i, line in enumerate(lines):
        # AQUI FOI MELHORADO: Verifica por "Total de Vulnerabilidades:" ou "Total de Ameaças:" ou "Total de Vulnerabilidades API:"
        if ("Total de Vulnerabilidades:" in line or "Total de Ameaças:" in line or "Total de Vulnerabilidades API:" in line) and not summary_line_found:
            summary_line = line
            summary_line_found = True
        else:
            parsed_content.append(line)

    if summary_line_found:
        parts = summary_line.split('|')
        for part in parts:
            part = part.strip()
            if "Total de Vulnerabilidades:" in part or "Total de Ameaças:" in part or "Total de Vulnerabilidades API:" in part:
                try:
                    summary["Total"] = int(part.split(':')[1].strip())
                except ValueError: pass
            elif "Críticas:" in part:
                try:
                    summary["Críticas"] = int(part.split(':')[1].strip())
                except ValueError: pass
            elif "Altas:" in part:
                try:
                    summary["Altas"] = int(part.split(':')[1].strip())
                except ValueError: pass
            elif "Médias:" in part:
                try:
                    summary["Médias"] = int(part.split(':')[1].strip())
                except ValueError: pass
            elif "Baixas:" in part:
                try:
                    summary["Baixas"] = int(part.split(':')[1].strip())
                except ValueError: pass
    
    return summary, "\n".join(parsed_content).strip()

def parse_raw_http_request(raw_request):
    """
    Parses a raw HTTP request string into method, path, headers, and body.
    Attempts to reconstruct a full URL from Host header and path.
    """
    method = ""
    path = ""
    full_url = ""
    headers = {}
    body = ""

    lines = raw_request.splitlines()
    
    # First line: METHOD /path?query HTTP/1.1
    if lines:
        first_line_parts = lines[0].split(' ')
        if len(first_line_parts) >= 2:
            method = first_line_parts[0].strip()
            path = first_line_parts[1].strip()

    header_started = False
    body_started = False
    for line in lines[1:]: # Start from the second line
        if not line.strip() and not body_started: # Empty line signifies end of headers, start of body
            body_started = True
            continue
        
        if not body_started: # Still parsing headers
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
        else: # Parsing body
            body += line + '\n'

    # Reconstruct full URL from Host header and path
    if 'Host' in headers and path:
        host = headers['Host']
        scheme = "https" if "443" in host or "https" in raw_request.lower().splitlines()[0] else "http" # Infer scheme
        if path.startswith('/') and urlparse(f"{scheme}://{host}").path != '/':
            full_url = f"{scheme}://{host}{path}"
        else:
             full_url = f"{scheme}://{host}{path}"

    return {
        "method": method,
        "path": path,
        "full_url": full_url,
        "headers": headers,
        "body": body.strip()
    }


# --- Funções das "Páginas" (TODAS DEFINIDAS AQUI, ANTES DO BLOCO PRINCIPAL QUE AS CHAMA) ---
# A ordem das funções importa no Python. Funções chamadas devem ser definidas antes.

def home_page():
    st.header("Bem-vindo ao SentinelAI - Plataforma de Segurança 🛡️")
    st.markdown("""
        Seu assistente de pentest e modelagem de ameaças, agora com inteligência artificial visual!
        Selecione uma opção na barra lateral para começar:
        - **Início**: Esta página.
        - **OWASP Vulnerability Details**: Digite uma falha ou categoria OWASP e obtenha detalhes completos.
        - **Análise de Requisições HTTP**: Cole uma requisição HTTP e identifique possíveis falhas de segurança.
        - **OWASP Image Analyzer**: Identifique vulnerabilidades OWASP em prints de tela ou imagens.
        - **Modelagem de Ameaças (STRIDE)**: Analise diagramas de arquitetura e identifique ameaças STRIDE.
        - **Pentest Lab**: Crie e teste mini-laboratórios de vulnerabilidades (em desenvolvimento).
        - **PoC Generator (HTML)**: Gere PoCs HTML para vulnerabilidades específicas.
        - **OpenAPI Analyzer**: Analise especificações de API em busca de falhas de segurança e melhorias de design.
        - **Static Code Analyzer**: Cole trechos de código para análise básica de segurança e busca por informações sensíveis.
        - **Search Exploit (NVD)**: Pesquise por CVEs e possíveis PoCs usando a API da NVD.
        - **Acunetix Scanner Insights**: Analise o output do Acunetix para insights e PoCs.
    """)
    st.info("Para começar, selecione uma das opções de análise na barra lateral.")


def modelagem_de_ameacas_page(llm_model_vision, llm_model_text):
    st.header("Modelagem de Ameaças (STRIDE) 📊")
    st.markdown("""
        Envie um diagrama de arquitetura (ou um print de tela) e uma descrição da sua aplicação.
        O SentinelAI irá analisar a imagem e o texto para identificar ameaças de segurança usando a metodologia STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege).
    """)

    # Variáveis já inicializadas globalmente

    def reset_stride_analysis():
        st.session_state.stride_image_uploaded = None
        st.session_state.stride_description_text = ""
        st.session_state.stride_analysis_result = ""
        st.session_state.stride_summary = None
        st.rerun()

    if st.button("Limpar e Fazer Nova Consulta", key="reset_stride_analysis_button"):
        reset_stride_analysis()

    # Upload do diagrama de arquitetura
    uploaded_diagram_file = st.file_uploader(
        "Selecione o diagrama de arquitetura (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="stride_file_uploader"
    )

    diagram_preview_placeholder = st.empty()

    if uploaded_diagram_file is not None:
        try:
            diagram_bytes = uploaded_diagram_file.getvalue() # Corrigido para uploaded_diagram_file
            diagram_img = Image.open(BytesIO(diagram_bytes))
            diagram_preview_placeholder.image(diagram_img, caption="Pré-visualização do Diagrama", use_container_width=True)
            st.session_state.stride_image_uploaded = diagram_img
        except Exception as e:
            st.error(f"Erro ao carregar o diagrama: {e}")
            st.session_state.stride_image_uploaded = None
    elif st.session_state.stride_image_uploaded:
        diagram_preview_placeholder.image(st.session_state.stride_image_uploaded, caption="Pré-visualização do Diagrama", use_container_width=True)
    else:
        st.session_state.stride_image_uploaded = None

    # Descrição da aplicação
    app_description = st.text_area(
        "Descreva a aplicação e sua arquitetura (componentes, fluxos de dados, etc.):",
        value=st.session_state.stride_description_text,
        placeholder="Ex: 'É um e-commerce com frontend React, backend Node.js, banco de dados MongoDB, e usa AWS S3 para armazenamento de imagens.'",
        height=150,
        key="stride_description_input"
    )
    st.session_state.stride_description_text = app_description.strip()

    if st.button("Analisar Arquitetura (STRIDE)", key="analyze_stride_button"):
        if st.session_state.stride_image_uploaded is None:
            st.error("Por favor, selecione um diagrama de arquitetura para análise.")
        elif not st.session_state.stride_description_text:
            st.error("Por favor, forneça uma descrição da aplicação e sua arquitetura.")
        else:
            with st.spinner("Realizando modelagem de ameaças STRIDE..."):
                stride_prompt = (
                    f"Você é um especialista em modelagem de ameaças e segurança de software, com profundo conhecimento na metodologia STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege).\n"
                    f"Sua tarefa é analisar o diagrama de arquitetura fornecido (na imagem) e a descrição da aplicação, e identificar ameaças de segurança usando o framework STRIDE.\n"
                    f"\n**RESUMO:** Forneça um resumo quantitativo na PRIMEIRA LINHA da sua resposta, no formato exato: `Total de Ameaças: X | Críticas: Y | Altas: Z | Médias: W | Baixas: V` (substitua X,Y,Z,W,V pelos números correspondentes). Se não houver ameaças, use 0.\n\n"
                    f"Para cada ameaça STRIDE identificada, forneça os seguintes detalhes de forma concisa e prática, utilizando formato Markdown para títulos e blocos de código:\n\n"
                    f"## Ameaça Identificada: [Nome da Ameaça]\n"
                    f"**Tipo STRIDE:** [S/T/R/I/D/E - Ex: Information Disclosure]\n"
                    f"**Severidade:** [Crítica/Alta/Média/Baixa]\n"
                    f"**Descrição:** Explique brevemente a ameaça e como ela se manifesta neste diagrama/descrição.\n"
                    f"**Árvore de Ataques (Simplificada):** Descreva os passos típicos que um atacante seguiria para explorar esta ameaça, como uma lista ou pequenos parágrafos, ilustrando o fluxo de ataque.\n"
                    f"**Impacto Potencial:** Qual o risco se esta ameaça for explorada?\n"
                    f"**Sugestão de Mitigação:** Ações concretas e específicas para mitigar esta ameaça, relevantes para a arquitetura apresentada. Seja direto e acionável.\n\n"
                    f"Se não encontrar ameaças óbvias, ou a informação for insuficiente, indique isso e sugira melhorias para a arquitetura ou para o diagrama/descrição.\n\n"
                    f"**Descrição da Aplicação/Arquitetura:**\n{st.session_state.stride_description_text}\n\n"
                    f"**Diagrama:** (Imagem anexada)"
                )

                stride_analysis_result_raw = obter_resposta_llm(llm_model_vision, [stride_prompt, st.session_state.stride_image_uploaded])

                if stride_analysis_result_raw:
                    # Chamar a função de parse e armazenar o resumo e o texto limpo
                    st.session_state.stride_summary, st.session_state.stride_analysis_result = parse_vulnerability_summary(stride_analysis_result_raw)
                else:
                    st.session_state.stride_analysis_result = "Não foi possível realizar a modelagem de ameaças. Tente refinar sua descrição ou diagrama."
                    st.session_state.stride_summary = None

    if st.session_state.stride_analysis_result:
        st.subheader("Resultados da Modelagem de Ameaças (STRIDE)")
        
        if st.session_state.stride_summary:
            st.markdown("#### Resumo das Ameaças Identificadas:")
            cols = st.columns(5)
            cols[0].metric("Total", st.session_state.stride_summary["Total"])
            cols[1].metric("Críticas", st.session_state.stride_summary["Críticas"])
            cols[2].metric("Altas", st.session_state.stride_summary["Altas"])
            cols[3].metric("Médias", st.session_state.stride_summary["Médias"])
            cols[4].metric("Baixas", st.session_state.stride_summary["Baixas"])
            st.markdown("---")
        
        st.markdown(st.session_state.stride_analysis_result)

def owasp_scout_visual_page(llm_model_vision, llm_model_text):
    st.header("OWASP Image Analyzer: Análise de Vulnerabilidades em Imagens 👁️")
    st.markdown("""
        Envie um print, um trecho de código em imagem, ou qualquer diagrama e pergunte ao SentinelAI se ele detecta vulnerabilidades OWASP Top 10.
        Quanto mais detalhes na sua pergunta, melhor a análise!
    """)

    # Variáveis já inicializadas globalmente

    def reset_owasp_scout_visual():
        st.session_state.owasp_image_uploaded = None
        st.session_state.owasp_question_text = ""
        st.session_state.owasp_analysis_result = ""
        st.session_state.owasp_consider_waf_state = False
        st.rerun() # Reexecuta para limpar a interface

    # Botão de Nova Consulta/Limpar
    if st.button("Limpar e Fazer Nova Consulta", key="reset_visual_analysis_button"):
        reset_owasp_scout_visual()

    # Elementos de entrada
    uploaded_file = st.file_uploader(
        "Selecione uma imagem para análise (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="owasp_file_uploader"
    )

    image_preview_placeholder = st.empty()

    if uploaded_file is not None:
        try:
            img_bytes = uploaded_file.getvalue()
            img = Image.open(BytesIO(img_bytes))
            image_preview_placeholder.image(img, caption="Pré-visualização da Imagem", use_container_width=True)
            st.session_state.owasp_image_uploaded = img
        except Exception as e:
            st.error(f"Erro ao carregar a imagem: {e}")
            st.session_state.owasp_image_uploaded = None
    elif st.session_state.owasp_image_uploaded:
        image_preview_placeholder.image(st.session_state.owasp_image_uploaded, caption="Pré-visualização da Imagem", use_container_width=True)
    else:
        st.session_state.owasp_image_uploaded = None


    question = st.text_area(
        "Sua pergunta sobre a vulnerabilidade ou contexto:",
        value=st.session_state.owasp_question_text, # Agora usa o valor do session_state
        placeholder="Ex: 'Esta tela de login é vulnerável?', 'Há XSS neste código?', 'Qual vulnerabilidade está presente neste diagrama?'",
        key="owasp_question_input"
    )
    st.session_state.owasp_question_text = question

    consider_waf = st.checkbox(
        "Considerar bypass de WAF?",
        value=st.session_state.owasp_consider_waf_state, # Agora usa o valor do session_state
        key="owasp_waf_checkbox"
    )
    # st.session_state.owasp_consider_waf_state = consider_waf # Não precisa atribuir de volta aqui

    if st.button("Analisar Vulnerabilidade", key="owasp_analyze_button_main"):
        if st.session_state.owasp_image_uploaded is None:
            st.error("Por favor, selecione uma imagem para análise.")
        elif not st.session_state.owasp_question_text:
            st.error("Por favor, digite sua pergunta sobre a vulnerabilidade na imagem.")
        else:
            with st.spinner("Analisando sua imagem em busca de vulnerabilidades OWASP..."):
                # --- PROMPT AJUSTADO PARA OWASP IMAGE ANALYZER (com Severidade e Dicas de Exploração) ---
                prompt_parts = [
                    f"Você é um especialista em segurança da informação e pentest."
                    f"Analise a imagem fornecida e a seguinte pergunta/contexto: '{st.session_state.owasp_question_text}'."
                    f"\n\nIdentifique possíveis vulnerabilidades de segurança da informação relevantes para a OWASP Top 10 (2021) que possam ser inferidas da imagem ou do contexto fornecido."
                    f"\n\nPara cada vulnerabilidade identificada, forneça os seguintes detalhes de forma concisa e prática, utilizando formato Markdown para títulos e blocos de código:"
                    f"\n\n## 1. Detalhamento da Falha"
                    f"\nUma breve explicação do que é a vulnerabilidade, como ela ocorre e os cenários comuns de impacto, **especificamente como se relaciona à imagem ou ao contexto.**"
                    f"\n\n## 2. Categoria OWASP (2021)"
                    f"\nIndique o CÓDIGO e o NOME da categoria da OWASP Top 10 (2021) à qual esta vulnerabilidade pertence (ex: A03: Injection). Use a lista: {', '.join([f'{c}: {n}' for c, n in OWASP_TOP_10_2021.items()])}. Se for uma subcategoria, mencione-la também."
                    f"\n\n## 3. Técnicas de Exploração Detalhadas"
                    f"\nDescreva passo a passo os métodos comuns e abordagens para testar e explorar esta vulnerabilidade, focando em como a imagem pode estar relacionada. Seja didático e prático.\n"
                    f"\n\n## 4. Ferramentas Sugeridas"
                    f"\nListe as ferramentas de segurança e pentest (ex: Burp Suite, Nmap, SQLmap, XSSer, Nessus, Nikto, Metasploit, etc.) que seriam úteis para descobrir e explorar esta vulnerabilidade, explicando brevemente como cada uma se aplicaria.\n"
                    f"\n\n## 5. Severidade"
                    f"\nClassifique a severidade desta vulnerabilidade: [Crítica/Alta/Média/Baixa].\n"
                    f"\n\n## 6. Dicas de Exploração / Próximos Passos Práticos"
                    f"\nCom base na falha identificada e no contexto da imagem, forneça dicas práticas e os próximos passos que um pentester faria para explorar ou confirmar a falha. Inclua instruções sobre como usar as ferramentas sugeridas e payloads de teste, se aplicável. Seja acionável.\n"
                ]
                
                if st.session_state.owasp_consider_waf_state:
                    prompt_parts.append(f"\n\n## 7. Dicas de Bypass de WAF")
                    prompt_parts.append(f"\nForneça estratégias, técnicas e exemplos práticos (se aplicável à vulnerabilidade) para contornar ou evadir a detecção de um Web Application Firewall (WAF) ao tentar explorar esta falha. Inclua exemplos de payloads ou modificações de requisições que podem ajudar a testar o presença ou bypass do WAF.")
                    poc_section_num = 8
                else:
                    poc_section_num = 7
                    
                prompt_parts.append(f"\n\n## {poc_section_num}. Prova de Conceito (PoC)")
                prompt_parts.append(f"\nForneça **exemplos práticos de comandos de terminal, requisições HTTP (com `curl` ou similar), ou payloads de código (Python, JS, etc.)** que demonstrem a exploração. Esses exemplos devem ser claros, prontos para uso (com pequenas adaptações) e encapsulados em blocos de código Markdown (` ``` `). Relacione o PoC à imagem ou contexto, se possível.")
                
                prompt_parts.append(f"\n\nSeu objetivo é ser direto, útil e focado em ações e informações completas para um pentester. Se a imagem não contiver vulnerabilidades óbvias, ou a pergunta for muito genérica, indique isso de forma clara.")
                # --- FIM DO PROMPT AJUSTADO ---

                full_prompt_list = [st.session_state.owasp_image_uploaded, "".join(prompt_parts)]

                analysis_result = obter_resposta_llm(llm_model_vision, full_prompt_list)
                
                if analysis_result:
                    st.session_state.owasp_analysis_result = analysis_result
                else:
                    st.session_state.owasp_analysis_result = "Não foi possível obter uma resposta do Gemini. Tente novamente."

    # Exibe o resultado da análise (se houver)
    if st.session_state.owasp_analysis_result:
        st.subheader("Resultados da Análise Visual") # Corrigido subtítulo
        st.markdown(st.session_state.owasp_analysis_result)

def owasp_text_analysis_page(llm_model_vision, llm_model_text):
    st.header("OWASP Vulnerability Details 📝") # NOME ATUALIZADO AQUI
    st.markdown("""
        Digite o CÓDIGO de uma categoria OWASP Top 10 (ex: `A03`) ou o NOME de uma falha específica (ex: `IDOR`, `XSS`, `SQL Injection`).
        O SentinelAI fornecerá detalhes completos sobre a vulnerabilidade.
    """)

    # Variáveis já inicializadas globalmente
    
    def reset_owasp_text_analysis():
        st.session_state.owasp_text_input_falha = ""
        st.session_state.owasp_text_analysis_result = ""
        st.session_state.owasp_text_context_input = ""
        st.session_state.owasp_text_consider_waf_state = False
        st.rerun()

    # Botão de Nova Consulta/Limpar para Análise por Texto
    if st.button("Limpar e Fazer Nova Consulta", key="reset_text_analysis_button"):
        reset_owasp_text_analysis()

    # Entrada do usuário para a falha/categoria
    user_input_falha = st.text_input(
        "Digite a falha ou categoria OWASP:",
        value=st.session_state.owasp_text_input_falha, # Agora usa o valor do session_state
        placeholder="Ex: A01, Injection, IDOR, Cross-Site Scripting",
        key="text_input_falha"
    )
    st.session_state.owasp_text_input_falha = user_input_falha.strip()


    # Contexto adicional para análise de texto
    contexto_texto = st.text_area(
        "Forneça um contexto adicional (opcional):",
        value=st.session_state.owasp_text_context_input, # Agora usa o valor do session_state
        placeholder="Ex: 'aplicação web em PHP', 'API REST com JWT', 'exploração via SQLi no parâmetro id'",
        key="text_context_input"
    )
    st.session_state.owasp_text_context_input = contexto_texto.strip()

    # Checkbox para WAF na análise de texto
    consider_waf_texto = st.checkbox(
        "Considerar bypass de WAF?",
        value=st.session_state.owasp_text_consider_waf_state, # Agora usa o valor do session_state
        key="text_consider_waf_checkbox"
    )
    # st.session_state.owasp_text_consider_waf_state = consider_waf_texto # Não precisa atribuir de volta aqui

    if st.button("Analisar Falha por Texto", key="analyze_text_button"):
        if not st.session_state.owasp_text_input_falha:
            st.error("Por favor, digite a falha ou categoria OWASP para análise.")
        else:
            categoria_owasp_codigo = None
            subcategoria_info = ""

            # Tenta mapear o input para uma categoria OWASP se não for um código direto
            if st.session_state.owasp_text_input_falha.upper() in OWASP_TOP_10_2021:
                categoria_owasp_codigo = st.session_state.owasp_text_input_falha.upper()
                st.info(f"Categoria OWASP selecionada: {OWASP_TOP_10_2021[categoria_owasp_codigo]}")
            else:
                categoria_owasp_codigo = mapear_falha_para_owasp(llm_model_text, st.session_state.owasp_text_input_falha)
                if categoria_owasp_codigo:
                    st.info(f"O LLM mapeou '{st.session_state.owasp_text_input_falha}' para a categoria OWASP: {OWASP_TOP_10_2021[categoria_owasp_codigo]}")
                    if categoria_owasp_codigo in OWASP_SUBCATEGORIES:
                        for sub in OWASP_SUBCATEGORIES[categoria_owasp_codigo]:
                            if st.session_state.owasp_text_input_falha.lower() in sub.lower():
                                subcategoria_info = f" Foco na subcategoria: **'{sub}'**."
                                break
                else:
                    st.error("Não foi possível identificar a categoria OWASP para a falha fornecida.")
                    st.session_state.owasp_text_analysis_result = ""
                    return

            if categoria_owasp_codigo:
                with st.spinner(f"Obtendo informações para {OWASP_TOP_10_2021[categoria_owasp_codigo]}..."):
                    prompt_base = (
                        f"Você é um especialista em segurança da informação e pentest."
                        f"Sua tarefa é fornecer informações detalhadas para a exploração da vulnerabilidade da OWASP Top 10 (2021) "
                        f"categorizada como **'{OWASP_TOP_10_2021[categoria_owasp_codigo]}' ({categoria_owasp_codigo})**."
                        f"\n\nPor favor, inclua os seguintes tópicos de forma concisa e prática, utilizando formato Markdown para títulos e blocos de código:"
                        f"\n\n## 1. Detalhamento da Falha"
                        f"\nUma breve explicação do que é a vulnerabilidade, como ela ocorre e os cenários comuns de impacto."
                        f"\n\n## 2. Técnicas de Exploração"
                        f"\nMétodos comuns e abordagens para testar e explorar esta vulnerabilidade."
                        f"\n\n## 3. Severidade e Impacto Técnico" # NOVO: Melhoria do título
                        f"\nClassifique a severidade desta vulnerabilidade: [Crítica/Alta/Média/Baixa].\n"
                        f"**Impacto Técnico:** Descreva o impacto técnico detalhado da exploração desta falha, com exemplos e consequências técnicas específicas.\n" # NOVO: Impacto Técnico
                        f"**CVSSv3.1 Score:** Forneça uma estimativa do score CVSS v3.1 para esta vulnerabilidade e o vetor CVSS. Ex: `7.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N)`\n" # NOVO: Score CVSS
                    )
                    
                    if st.session_state.owasp_text_consider_waf_state:
                        prompt_base += f"\n\n## 4. Dicas de Bypass de WAF"
                        prompt_base += f"\nForneça estratégias, técnicas e exemplos práticos (se aplicável à vulnerabilidade) para contornar ou evadir a detecção de um Web Application Firewall (WAF) ao tentar explorar esta falha. Inclua exemplos de payloads ou modificações de requisições que podem ajudar a testar o presença ou bypass do WAF."
                        poc_section_num = 5
                        solution_section_num = 6
                        benefits_risks_section_num = 7
                    else:
                        poc_section_num = 4
                        solution_section_num = 5
                        benefits_risks_section_num = 6
                        
                    prompt_base += (
                        f"\n\n## {poc_section_num}. Prova de Conceito (PoC)"
                        f"\nForneça **exemplos práticos de comandos de terminal, requisições HTTP (com `curl` ou similar), ou payloads de código (Python, JS, etc.)** que demonstrem a exploração. Esses exemplos devem ser claros, prontos para uso (com pequenas adaptações) e encapsulados em blocos de código Markdown (` ``` `)."
                        f"\n\n## {solution_section_num}. Detalhamento da Solução"
                        f"\nDescreva as ações recomendadas para corrigir o vulnerabilidade de forma eficaz."
                        f"\n\n## {benefits_risks_section_num}. Benefícios e Riscos da Correção"
                        f"\nQuais são os benefícios de implementar a solução e os possíveis riscos ou impactos colaterais da sua aplicação?"
                        f"\n\nSeu objetivo é ser direto, útil e focado em ações e informações completas para um pentester, como um resumo para um relatório de pentest."
                    )

                    analysis_result = obter_resposta_llm(llm_model_text, [prompt_base])
                    
                    if analysis_result:
                        st.session_state.owasp_text_analysis_result = analysis_result
                    else:
                        st.session_state.owasp_text_analysis_result = "Não foi possível obter uma resposta do Gemini. Tente novamente."
            else:
                st.error("Não foi possível identificar a categoria OWASP para a falha fornecida.")
                st.session_state.owasp_text_analysis_result = ""

    # Exibe o resultado da análise de texto (se houver)
    if st.session_state.owasp_text_analysis_result:
        st.subheader("Resultados da Análise por Texto")
        st.markdown(st.session_state.owasp_text_analysis_result)

def http_request_analysis_page(llm_model_vision, llm_model_text):
    st.header("Análise de Requisições HTTP 📡")
    st.markdown("""
        Cole a URL alvo e a requisição HTTP completa (RAW) do Burp Suite ou similar.
        O SentinelAI irá analisar a requisição em busca de **múltiplas falhas de segurança OWASP Top 10**, incluindo:
        - Injeções (SQLi, XSS, Command, etc.)
        - Falhas de autenticação/sessão
        - Configurações incorretas (headers, métodos HTTP, etc.)
        - Exposição de dados sensíveis
        - Falhas de controle de acesso
        - SSRF e outros tipos de falhas em componentes externos
        E sugerir **Provas de Conceito (PoCs) acionáveis** para testar essas falhas.
    """)

    # Variáveis já inicializadas globalmente
    
    def reset_http_request_analysis():
        st.session_state.http_request_input_url = ""
        st.session_state.http_request_input_raw = ""
        st.session_state.http_request_analysis_result = ""
        st.session_state.http_request_consider_waf_state = False
        st.session_state.http_request_summary = None
        st.rerun()

    if st.button("Limpar e Fazer Nova Consulta", key="reset_http_request_button"):
        reset_http_request_analysis()

    # Campo para a URL/TARGET
    target_url = st.text_input(
        "URL Alvo (Target):",
        value=st.session_state.http_request_input_url,
        placeholder="Ex: https://testphp.vulnweb.com/search.php",
        key="http_request_target_url_input"
    )
    st.session_state.http_request_input_url = target_url.strip()

    # Entrada do usuário para a requisição HTTP
    http_request_raw = st.text_area(
        "Cole a requisição HTTP RAW aqui:",
        value=st.session_state.http_request_input_raw,
        placeholder="Ex: POST /search.php?... HTTP/1.1\nHost: ...\nContent-Length: ...",
        height=300,
        key="http_request_input_area"
    )
    st.session_state.http_request_input_raw = http_request_raw.strip()

    # Checkbox para WAF na análise de requisição
    consider_waf_http = st.checkbox(
        "Considerar bypass de WAF?",
        value=st.session_state.http_request_consider_waf_state,
        key="http_request_waf_checkbox"
    )
    # st.session_state.http_request_consider_waf_state = consider_waf_http # Não precisa atribuir de volta aqui

    if st.button("Analisar Requisição", key="analyze_http_request_button"):
        if not st.session_state.http_request_input_url:
            st.error("Por favor, forneça a URL Alvo para análise.")
        elif not st.session_state.http_request_input_raw:
            st.error("Por favor, cole a requisição HTTP RAW para análise.")
        else:
            with st.spinner("Analisando a requisição HTTP com LLM..."):
                # Parse da requisição RAW para extrair componentes (já existe e funciona bem)
                parsed_request = parse_raw_http_request(st.session_state.http_request_input_raw)
                
                # Adaptação para o prompt, garantindo que mesmo se o parse não for perfeito, o LLM ainda veja o RAW completo
                request_method_path_version = f"{parsed_request['method']} {parsed_request['path']} HTTP/1.1" if parsed_request['method'] and parsed_request['path'] else "Não detectado"
                headers_formatted = "\n".join([f"{k}: {v}" for k, v in parsed_request['headers'].items()])
                body_content = parsed_request['body']

                prompt_base = (
                    f"Você é um especialista em segurança da informação e pentest. Analise a requisição HTTP RAW fornecida e a URL alvo. Identifique **TODAS as possíveis falhas de segurança OWASP Top 10 (2021) e outras vulnerabilidades relevantes aplicáveis**, sendo extremamente detalhado e preciso na análise de cada parte da requisição. Inclua:\n"
                    f"\n**RESUMO:** Forneça um resumo quantitativo na PRIMEIRA LINHA da sua resposta, no formato exato: `Total de Vulnerabilidades: X | Críticas: Y | Altas: Z | Médias: W | Baixas: V` (substitua X,Y,Z,W,V pelos números correspondentes). Se não houver vulnerabilidades, use 0.\n\n"
                    f"Para cada **falha potencial** identificado, apresente de forma concisa e prática:\n\n"
                    f"1.  **Tipo da Falha e Categoria OWASP (2021):** Ex: `Injeção SQL (A03: Injection)` ou `Exposição de Cookie Sensível`.\n"
                    f"2.  **Detalhes e Impacto:** Breve descrição da falha e como ela pode ser explorada nesta requisição específica, mencionando qual parte da requisição (linha, cabeçalho, corpo) está envolvida.\n"
                    f"3.  **Severidade:** [Crítica/Alta/Média/Baixa]\n"
                    f"4.  **Prova de Conceito (PoC) - REQUISIÇÃO HTTP RAW COMPLETA MODIFICADA:** Forneça **A REQUISIÇÃO HTTP RAW COMPLETA MODIFICADA** que demonstre a exploração da falha. Esta requisição RAW deve ser pronta para ser copiada e colada em um proxy (como Burp Suite Repeater) ou enviada via `netcat`. Encapsule a requisição RAW completa em um bloco de código Markdown com a linguagem `http` (` ```http `). Certifique-se de que a PoC é funcional e reflete a exploração da vulnerabilidade.\n"
                    f"5.  **Ferramentas Sugeridas:** Liste ferramentas de segurança e pentest (ex: Burp Suite, Nmap, SQLmap, XSSer, Nessus, Nikto, Metasploit, dirbuster, ffuf, ZAP, etc.) que seriam úteis para descobrir e explorar esta vulnerabilidade, explicando brevemente como cada uma se aplicaria.\n"
                )

                if st.session_state.http_request_consider_waf_state:
                    # Seção de WAF fica após os pontos gerais e antes da análise segmentada
                    prompt_base += f"\n\n6.  **Dicas de Bypass de WAF:** Se a requisição tiver um WAF, inclua estratégias e exemplos de bypass nos PoCs (na própria requisição RAW modificada, se possível), se aplicável. Inclua técnicas como obfuscação, codificação alternativa, uso de múltiplos headers, etc.\n"
                
                prompt_base += f"\n\n--- Análise Segmentada Detalhada da Requisição ---\n"
                prompt_base += f"**URL Alvo Fornecida:** `{st.session_state.http_request_input_url}`\n"
                prompt_base += f"**Requisição RAW Original Completa:**\n```http\n{st.session_state.http_request_input_raw}\n```\n"

                prompt_base += f"\n### 1. Análise da Linha de Requisição (`{request_method_path_version}`):\n"
                prompt_base += f"Foque em:\n"
                prompt_base += f"- Possíveis injeções (SQLi, XSS, Command, Path Traversal) em parâmetros de URL.\n"
                prompt_base += f"- Verbos HTTP inadequados ou métodos não permitidos.\n"
                prompt_base += f"- Exposição de informações sensíveis no path ou em parâmetros de query.\n"
                f"- Falhas relacionadas à versão HTTP (ex: `HTTP/1.1`) como HTTP Request Smuggling (Desync). Analise a possibilidade de adicionar `Transfer-Encoding: chunked` para tentar desync. \n"
                
                prompt_base += f"\n### 2. Análise dos Cabeçalhos HTTP (`{headers_formatted}`):\n"
                prompt_base += f"Foque em:\n"
                f"- Falhas de segurança como CSRF (verificar `Origin`, `Referer`), Clickjacking (`X-Frame-Options`).\n"
                f"- Cabeçalhos de segurança ausentes ou incorretos (HSTS, CSP, X-Content-Type-Options, Referrer-Policy, etc.).\n"
                f"- Manipulação de cookies: Verifique atributos de segurança (HTTPOnly, Secure, SameSite, `Max-Age`/`Expires` para cookies de sessão). **Importante: Analise o valor dos cookies (ex: `wordpress_test_cookie=WP%20Cookie%20check`). Se houver valores como JWT, MD5, ou outros hashes/tokens, mencione a possível exposição ou vulnerabilidade.**\n"
                f"- Problemas de cache (Cache Poisoning).\n"
                f"- Bypass de controle de acesso via cabeçalhos (ex: `X-Forwarded-For`, `X-Original-URL`).\n"
                f"- Host Header Injection.\n"
                f"- Falhas de CORS (Cross-Origin Resource Sharing) em cabeçalhos como `Origin`, `Access-Control-Allow-Origin`.\n"
                f"- Exposição de informações no `User-Agent` ou `Referer`.\n"
                
                prompt_base += f"\n### 3. Análise do Corpo da Requisição (`{body_content}`):\n"
                prompt_base += f"Foque em:\n"
                f"- Injeções (SQLi, XSS, Command, NoSQL, XML, JSON) em dados enviados.\n"
                f"- Desserialização insegura (se o corpo contiver dados serializados).\n"
                f"- Vulnerabilidades em upload de arquivos (se for um request de upload).\n"
                f"- Bypass de validação de input ou lógica de negócio.\n"

                prompt_base += (
                    f"\nSe a requisição não contiver vulnerabilidades óbvias que possam ser exploradas directamente, indique isso de forma clara e sugira o que mais o pentester poderia investigar com base nesta requisição (ex: outras rotas, enumeração de diretórios, análise de respostas para informações sensíveis, fuzzing de parâmetros, análise de erros, etc.).\n\n"
                    f"Sua resposta deve ser direta, útil e focada em ações e informações completas para um pentester, apresentando cada falha identificada com seus detalhes, ferramentas e PoC completos."
                )
                
                analysis_result_raw = obter_resposta_llm(llm_model_text, [prompt_base])
                
                if analysis_result_raw:
                    st.session_state.http_request_summary, st.session_state.http_request_analysis_result = parse_vulnerability_summary(analysis_result_raw)
                else:
                    st.session_state.http_request_analysis_result = "Não foi possível obter uma resposta do Gemini. Tente novamente."
                    st.session_state.http_request_summary = None

    if st.session_state.http_request_analysis_result:
        st.subheader("Resultados da Análise de Requisições HTTP")
        
        if st.session_state.http_request_summary:
            st.markdown("#### Resumo das Vulnerabilidades Identificadas:")
            cols = st.columns(5)
            cols[0].metric("Total", st.session_state.http_request_summary["Total"])
            cols[1].metric("Críticas", st.session_state.http_request_summary["Críticas"])
            cols[2].metric("Altas", st.session_state.http_request_summary["Altas"])
            cols[3].metric("Médias", st.session_state.http_request_summary["Médias"])
            cols[4].metric("Baixas", st.session_state.http_request_summary["Baixas"])
            st.markdown("---")

        st.markdown(st.session_state.http_request_analysis_result)

def pentest_lab_page(llm_model_vision, llm_model_text):
    st.header("Pentest Lab: Seu Laboratório de Vulnerabilidades 🧪")
    st.markdown("""
        Selecione uma vulnerabilidade e o SentinelAI irá gerar um mini-laboratório HTML básico (PoC em HTML) para que você possa testar a falha diretamente no seu navegador.
        Também fornecerá dicas de como explorar e o payload/comando para o teste.
        **AVISO: Este laboratório é para fins educacionais e de teste. Não execute payloads em sites reais.**
    """)

    # Variáveis já inicializadas globalmente
    
    def reset_pentest_lab():
        st.session_state.lab_vulnerability_selected = None
        st.session_state.lab_html_poc = ""
        st.session_state.lab_explanation = ""
        st.session_state.lab_payload_example = ""
        st.rerun()

    if st.button("Limpar Laboratório", key="reset_lab_button"):
        reset_pentest_lab()

    # Seleção da Vulnerabilidade
    vulnerability_options = ["Escolha uma vulnerabilidade"] + sorted(OWASP_SUBCATEGORIES["A03"]) # Foco em injeções para HTML PoC
    
    selected_vuln = st.selectbox(
        "Selecione a vulnerabilidade para o laboratório:",
        options=vulnerability_options,
        index=0,
        key="lab_vuln_select"
    )
    st.session_state.lab_vulnerability_selected = selected_vuln if selected_vuln != "Escolha uma vulnerabilidade" else None

    if st.button("Gerar Laboratório", key="generate_lab_button"):
        if not st.session_state.lab_vulnerability_selected:
            st.error("Por favor, selecione uma vulnerabilidade para gerar o laboratório.")
        else:
            with st.spinner(f"Gerando laboratório para {st.session_state.lab_vulnerability_selected}..."):
                # Prompt para gerar o mini-laboratório HTML e a explicação
                lab_prompt = (
                    f"Você é um especialista em pentest e educador. Sua tarefa é criar um mini-laboratório HTML simples e um payload para demonstrar a vulnerabilidade '{st.session_state.lab_vulnerability_selected}'.\n"
                    f"Forneça as informações nos seguintes tópicos:\n\n"
                    f"## 1. Descrição da Vulnerabilidade e Dicas de Exploração\n"
                    f"Uma breve explicação do que é a vulnerabilidade, como ela funciona e dicas práticas de como tentar explorá-la.\n\n"
                    f"## 2. Mini-Laboratório HTML (PoC HTML)\n"
                    f"Forneça um **código HTML COMPLETO e MÍNIMO** (com tags `<html>`, `<head>`, `<body>`) que simule um cenário vulnerável a **{st.session_state.lab_vulnerability_selected}**.\n"
                    f"Este HTML deve ser funcional e auto-contido. O foco é na vulnerabilidade, não no design.\n"
                    f"Encapsule o HTML completo em um bloco de código Markdown com a linguagem `html` (` ```html `).\n\n"
                    f"## 3. Exemplo de Payload/Comando para Teste\n"
                    f"Forneça o payload ou comando específico que o usuário injetaria ou usaria neste HTML para provar a vulnerabilidade. Encapsule em um bloco de código Markdown com la linguagem apropriada (ex: ` ```js `, ` ```sql `, ` ```bash `).\n"
                    f"Este payload deve ser adaptado para o HTML gerado no PoC HTML.\n"
                    f"\nSeja didático e direto. O objetivo é que o usuário possa copiar e colar o HTML e o payload para testar."
                )

                lab_generation_raw = obter_resposta_llm(llm_model_text, [lab_prompt])
                
                if lab_generation_raw:
                    st.session_state.lab_explanation = lab_generation_raw
                    
                    # Tenta extrair o HTML e o payload
                    html_start = lab_generation_raw.find("```html")
                    html_end = lab_generation_raw.find("```", html_start + len("```html"))
                    
                    payload_start_marker = "```"
                    
                    if html_start != -1 and html_end != -1:
                        payload_start = lab_generation_raw.find(payload_start_marker, html_end + 1)
                    else:
                        payload_start = lab_generation_raw.find(payload_start_marker)
                        
                    payload_end = -1
                    if payload_start != -1:
                        payload_end = lab_generation_raw.find(payload_start_marker, payload_start + len(payload_start_marker))
                        if payload_end == payload_start:
                            payload_end = -1

                    if html_start != -1 and html_end != -1:
                        st.session_state.lab_html_poc = lab_generation_raw[html_start + len("```html") : html_end].strip()
                    else:
                        st.session_state.lab_html_poc = "Não foi possível extrair o HTML do laboratório. Verifique a resposta do LLM."
                    
                    if payload_start != -1 and payload_end != -1:
                        payload_content = lab_generation_raw[payload_start + len(payload_start_marker) : payload_end].strip()
                        if '\n' in payload_content and payload_content.splitlines()[0].strip().isalpha():
                            st.session_state.lab_payload_example = '\n'.join(payload_content.splitlines()[1:]).strip()
                        else:
                            st.session_state.lab_payload_example = payload_content
                    else:
                        st.session_state.lab_payload_example = "Não foi possível extrair o exemplo de payload. Verifique a resposta do LLM."
                else:
                    st.session_state.lab_explanation = "Não foi possível gerar o laboratório para a vulnerabilidade selecionada."
                    st.session_state.lab_html_poc = ""
                    st.session_state.lab_payload_example = ""

    if st.session_state.lab_html_poc or st.session_state.lab_explanation:
        st.subheader("Resultados do Laboratório") # Corrigido subtítulo
        
        st.markdown(st.session_state.lab_explanation)

        if st.session_state.lab_html_poc:
            st.markdown("#### Mini-Laboratório HTML (Copie e Cole em um arquivo .html e abra no navegador)")
            st.code(st.session_state.lab_html_poc, language="html")
            
            st.markdown("---")
            st.markdown("#### Teste o Laboratório Aqui (Visualização Direta)")
            st.warning("AVISO: Esta visualização direta é para conveniência. Para um teste real e isolado, **salve o HTML em um arquivo .html e abra-o diretamente no seu navegador**.")
            components.html(st.session_state.lab_html_poc, height=300, scrolling=True)
            st.markdown("---")

        if st.session_state.lab_payload_example: # Corrigido para poc_gen_payload_example
            st.markdown("#### Exemplo de Payload/Comando para Teste")
            payload_lang = "plaintext"
            first_line = st.session_state.lab_payload_example.splitlines()[0].strip() if st.session_state.lab_payload_example else ""
            
            if "alert(" in st.session_state.lab_payload_example.lower() or "document.write" in st.session_state.lab_payload_example.lower():
                payload_lang = "js"
            elif "SELECT " in st.session_state.lab_payload_example.upper() and "FROM " in st.session_state.lab_payload_example.upper():
                payload_lang = "sql"
            elif "http" in first_line.lower() and ("post" in first_line.lower() or "get" in first_line.lower()):
                payload_lang = "http"
            elif "curl " in first_line.lower() or "bash" in first_line.lower():
                payload_lang = "bash"
            elif "python" in first_line.lower() or "import" in st.session_state.lab_payload_example.lower():
                payload_lang = "python"
            
            st.code(st.session_state.lab_payload_example, language=payload_lang)


# --- Nova Página: PoC Generator (HTML) ---
def poc_generator_html_page(llm_model_vision, llm_model_text):
    st.header("PoC Generator (HTML): Crie Provas de Conceito em HTML 📄")
    st.markdown("""
        Gere códigos HTML de Prova de Conceito para testar vulnerabilidades específicas no navegador.
        Perfect para demonstrar falhas como CSRF, Clickjacking, CORS, e XSS baseados em HTML.
    """)

    # Variáveis já inicializadas globalmente
    
    def reset_poc_generator():
        st.session_state.poc_gen_vulnerability_input = ""
        st.session_state.poc_gen_context_input = ""
        st.session_state.poc_gen_html_output = ""
        st.session_state.poc_gen_instructions = ""
        st.session_state.poc_gen_payload_example = ""
        st.rerun()

    if st.button("Limpar Gerador", key="reset_poc_gen_button"):
        reset_poc_generator()

    vulnerability_input = st.text_input(
        "Digite a vulnerabilidade para gerar a PoC HTML (Ex: CSRF, Clickjacking, CORS, XSS):",
        value=st.session_state.poc_gen_vulnerability_input, # Agora usa o valor do session_state
        placeholder="Ex: CSRF, Clickjacking, CORS, XSS refletido",
        key="poc_gen_vuln_input"
    )
    st.session_state.poc_gen_vulnerability_input = vulnerability_input.strip()

    context_input = st.text_area(
        "Contexto Adicional (URL alvo, parâmetros, método, etc.):",
        value=st.session_state.poc_gen_context_input, # Agora usa o valor do session_state
        placeholder="Ex: 'URL: https://exemplo.com/transferencia, Parâmetros: conta=123&valor=100, Método: POST'",
        key="poc_gen_context_input_area"
    )
    st.session_state.poc_gen_context_input = context_input.strip()

    if st.button("Gerar PoC HTML", key="generate_poc_html_button"):
        if not st.session_state.poc_gen_vulnerability_input:
            st.error("Por favor, digite a vulnerabilidade para gerar a PoC.")
        else:
            with st.spinner(f"Gerando PoC HTML para {st.session_state.poc_gen_vulnerability_input}..."):
                poc_prompt = (
                    f"Você é um especialista em pentest e possui autorização para realizar testes de segurança. "
                    f"Sua tarefa é gerar uma Prova de Conceito (PoC) em HTML funcional e um payload/instruções para demonstrar a vulnerabilidade '{st.session_state.poc_gen_vulnerability_input}'.\n"
                    f"**Contexto:** {st.session_state.poc_gen_context_input if st.session_state.poc_gen_context_input else 'Nenhum contexto adicional fornecido.'}\n\n"
                    f"Forneça as informações nos seguintes tópicos:\n\n"
                    f"## 1. Detalhes da Vulnerabilidade e Como Funciona\n"
                    f"Uma breve explicação de como a vulnerabilidade funciona e como a PoC a demonstra.\n\n"
                    f"## 2. Código HTML da PoC (Completo e Mínimo)\n"
                    f"Forneça um **código HTML COMPLETO e MÍNIMO** (com tags `<html>`, `<head>`, `<body>`) que simule um cenário vulnerável a **{st.session_state.poc_gen_vulnerability_input}**.\n"
                    f"Este HTML deve ser funcional e auto-contido. O foco é na vulnerabilidade, não no design.\n"
                    f"Encapsule o HTML completo em um bloco de código Markdown com a linguagem `html` (` ```html `).\n\n"
                    f"## 3. Instruções de Uso e Payload (se aplicável)\n"
                    f"Descreva como o usuário deve usar este HTML para testar a PoC. Se for necessário um payload ou comando específico (ex: um script XSS, uma URL modificada para Clickjacking), forneça-o explicitamente e encapsule-o em um bloco de código Markdown com la linguagem apropriada (ex: ` ```js `, ` ```sql `, ` ```bash `, ` ```http `).\n"
                    f"\nSeja direto, prático e didático. O objetivo é que o usuário (um pentester autorizado) possa copiar e colar o HTML e as instruções para testar a falha em um ambiente de teste autorizado."
                )

                poc_generation_raw = obter_resposta_llm(llm_model_text, [poc_prompt])

                if poc_generation_raw:
                    st.session_state.poc_gen_instructions = poc_generation_raw
                    
                    html_start = poc_generation_raw.find("```html")
                    html_end = poc_generation_raw.find("```", html_start + len("```html"))
                    
                    payload_start_marker = "```"
                    
                    if html_start != -1 and html_end != -1:
                        payload_start = poc_generation_raw.find(payload_start_marker, html_end + 1)
                    else:
                        payload_start = poc_generation_raw.find(payload_start_marker)
                        
                    payload_end = -1
                    if payload_start != -1:
                        payload_end = poc_generation_raw.find(payload_start_marker, payload_start + len(payload_start_marker))
                        if payload_end == payload_start:
                            payload_end = -1

                    if html_start != -1 and html_end != -1:
                        st.session_state.poc_gen_html_output = poc_generation_raw[html_start + len("```html") : html_end].strip()
                    else:
                        st.session_state.poc_gen_html_output = "Não foi possível extrair o HTML da PoC. Verifique a resposta do LLM."
                    
                    if payload_start != -1 and payload_end != -1:
                        payload_content = poc_generation_raw[payload_start + len(payload_start_marker) : payload_end].strip()
                        if '\n' in payload_content and payload_content.splitlines()[0].strip().isalpha():
                            st.session_state.poc_gen_payload_example = '\n'.join(payload_content.splitlines()[1:]).strip()
                        else:
                            st.session_state.poc_gen_payload_example = payload_content
                    else:
                        st.session_state.poc_gen_payload_example = "Não foi possível extrair o exemplo de payload. Verifique a resposta do LLM."
                else:
                    st.session_state.poc_gen_instructions = "Não foi possível gerar a PoC HTML para a vulnerabilidade selecionada."
                    st.session_state.poc_gen_html_output = ""
                    st.session_state.poc_gen_payload_example = ""

    if st.session_state.poc_gen_html_output or st.session_state.poc_gen_instructions:
        st.subheader("Results da PoC HTML") # Corrigido subtítulo
        
        st.markdown(st.session_state.poc_gen_instructions)

        if st.session_state.poc_gen_html_output:
            st.markdown("#### Código HTML da PoC (Copie e Cole em um arquivo .html)")
            st.code(st.session_state.poc_gen_html_output, language="html")
            
            st.markdown("---")
            st.markdown("#### Teste a PoC Aqui (Visualização Direta)")
            st.warning("AVISO: Esta visualização direta é para conveniência. Para um teste real e isolado, **salve o HTML em um arquivo .html e abra-o diretamente no seu navegador**.")
            components.html(st.session_state.poc_gen_html_output, height=300, scrolling=True)
            st.markdown("---")

        if st.session_state.poc_gen_payload_example: # Corrigido para poc_gen_payload_example
            st.markdown("#### Exemplo de Payload/Comando para Teste")
            payload_lang = "plaintext"
            first_line = st.session_state.poc_gen_payload_example.splitlines()[0].strip() if st.session_state.poc_gen_payload_example else ""
            
            if "alert(" in st.session_state.poc_gen_payload_example.lower() or "document.write" in st.session_state.poc_gen_payload_example.lower():
                payload_lang = "js"
            elif "SELECT " in st.session_state.poc_gen_payload_example.upper() and "FROM " in st.session_state.poc_gen_payload_example.upper():
                payload_lang = "sql"
            elif "http" in first_line.lower() and ("post" in first_line.lower() or "get" in first_line.lower()):
                payload_lang = "http"
            elif "curl " in first_line.lower() or "bash" in first_line.lower():
                payload_lang = "bash"
            elif "python" in first_line.lower() or "import" in st.session_state.poc_gen_payload_example.lower():
                payload_lang = "python"
            
            st.code(st.session_state.poc_gen_payload_example, language=payload_lang)


# --- Nova Página: Static Code Analyzer (Basic) ---
def static_code_analyzer_page(llm_model_vision, llm_model_text):
    st.header("Static Code Analyzer (Basic) 👨‍💻")
    st.markdown("""
        Cole um trecho de código para análise básica de segurança. O SentinelAI irá identificar
        vulnerabilidades comuns (OWASP Top 10), padrões de exposição de informações sensíveis
        (chaves, IPs, comentários) e sugerir correções e PoCs.
        **AVISO:** Esta é uma análise de *primeira linha* e não substitui um SAST completo.
    """)

    # Inicializa ou reseta o estado
    if 'code_input_content' not in st.session_state:
        st.session_state.code_input_content = ""
    if 'code_analysis_result' not in st.session_state:
        st.session_state.code_analysis_result = ""
    if 'code_language_selected' not in st.session_state:
        st.session_state.code_language_selected = "Python" # Default

    def reset_code_analyzer():
        st.session_state.code_input_content = ""
        st.session_state.code_analysis_result = ""
        st.session_state.code_language_selected = "Python"
        st.rerun()

    if st.button("Limpar Análise de Código", key="reset_code_analysis_button"):
        reset_code_analyzer()

    # Campo para colar o código
    code_content = st.text_area(
        "Cole o trecho de código aqui:",
        value=st.session_state.code_input_content,
        placeholder="Ex: import os\napi_key = 'YOUR_SECRET_KEY'\ndef query_db(user_input):\n  conn = sqlite3.connect('app.db')\n  cursor = conn.cursor()\n  cursor.execute(f\"SELECT * FROM users WHERE username = '{user_input}'\")",
        height=300,
        key="code_input_area"
    )
    st.session_state.code_input_content = code_content.strip()

    # Seleção da Linguagem
    language_options = ["Python", "JavaScript", "Java", "PHP", "Go", "Ruby", "C#", "SQL", "Outra"]
    selected_language = st.selectbox(
        "Linguagem do Código:",
        options=language_options,
        index=language_options.index(st.session_state.code_language_selected),
        key="code_language_select"
    )
    st.session_state.code_language_selected = selected_language

    if st.button("Analisar Código", key="analyze_code_button"):
        if not st.session_state.code_input_content:
            st.error("Por favor, cole o código para análise.")
        else:
            with st.spinner(f"Analisando código {st.session_state.code_language_selected} com LLM..."):
                code_prompt = (
                    f"Você é um especialista em segurança de código e pentest. Analise o trecho de código fornecido na linguagem {st.session_state.code_language_selected}. "
                    f"Seu objetivo é identificar **TODAS as potenciais vulnerabilidades de segurança** (baseadas na OWASP Top 10 e outras falhas comuns) e **exposição de informações sensíveis**, tais como:\n"
                    f"- Chaves de API, chaves secretas ou tokens (ex: `API_KEY`, `secret_key`, `token`, `password`)\n"
                    f"- Endereços IP de servidores ou URLs internas/de desenvolvimento (ex: `192.168.1.1`, `dev.api.internal`, `test.database.com`)\n"
                    f"- Comentários de desenvolvedores que possam conter informações sensíveis (ex: `TODO: remover esta senha`, `FIXME: credenciais hardcoded aqui`, `username: admin / password: 123`)\n"
                    f"- Nomes de diretórios ou caminhos de arquivos internos/sensíveis (ex: `/var/www/backup`, `/admin/dev_tools`, `C:\\secrets\\config.ini`)\n\n"
                    f"**Código para análise:**\n```\n{st.session_state.code_input_content}\n```\n\n"
                    f"Para cada **achado (vulnerabilidade ou informação sensível)** identificado, apresente de forma concisa e prática, utilizando Markdown:\n\n"
                    f"## [Tipo de Achado (Ex: Injeção SQL, Chave de API Exposta, Credenciais em Comentário)]\n"
                    f"**Categoria OWASP (se aplicável):** [Ex: A03: Injection, A05: Security Misconfiguration]. Se for uma informação sensível não OWASP, indique 'Exposição de Informação'.\n"
                    f"**Severidade/Risco:** [Crítica/Alta/Média/Baixa - explique o impacto deste achado específico, tanto para vulnerabilidades quanto para informações expostas]\n"
                    f"**Detalhes no Código:** Explique onde no código a falha/informação foi observada. Inclua o **número da linha aproximado** se possível. Ex: `Linha 5: O parâmetro 'username' é usado diretamente em uma query SQL.`\n"
                    f"**Trecho de Código Afetado:** Forneça o trecho de código exato que contém a falha ou informação sensível. Encapsule-o em um bloco de código Markdown com a linguagem correspondente (ex: ```python, ```javascript, ```java). Este trecho deve ser facilmente identificável no código original.\n\n"
                    f"**Exemplo de PoC/Cenário de Exploração (se aplicável):** Descreva os passos para explorar a vulnerabilidade ou o risco de exposição da informação. Forneça exemplos de payloads, comandos ou trechos de código que demonstrem o problema. Para informações sensíveis, explique como essa exposição pode ser explorada (ex: acesso a sistemas, reconhecimento, pivotagem).\n"
                    f"Encapsule os exemplos de código em blocos de código Markdown (` ```{st.session_state.code_language_selected} ` ou ` ```bash `).\n\n"
                    f"**Ferramentas Sugeridas (se aplicável):** Liste ferramentas que podem ser usadas para explorar ou validar este achado. (Ex: `grep` para buscas de strings, `curl` para testar URLs, `nuclei` para templates, Burp Suite, etc.).\n\n"
                    f"**Recomendação/Mitigação:** Ações concretas para corrigir o problema ou mitigar o risco (ex: usar prepared statements, sanitizar input, remover hardcoded secrets, usar variáveis de ambiente, configurar permissões adequadas).\n\n"
                    f"Se não encontrar vulnerabilidades óbvias ou informações sensíveis, indique isso claramente. Lembre-se, sua análise é uma *primeira linha* e não substitui um SAST completo ou uma revisão de código manual profunda.\n\n"
                )

                code_analysis_result_raw = obter_resposta_llm(llm_model_text, [code_prompt])
                
                if code_analysis_result_raw:
                    st.session_state.code_analysis_result = code_analysis_result_raw
                else:
                    st.session_state.code_analysis_result = "Não foi possível obter a análise de código. Tente novamente."

    if st.session_state.code_analysis_result:
        st.subheader("Results da Análise de Código") # Corrigido subtítulo
        st.markdown(st.session_state.code_analysis_result)


# --- Nova Página: OpenAPI Analyzer ---
def swagger_openapi_analyzer_page(llm_model_vision, llm_model_text):
    st.header("OpenAPI Analyzer: Análise de APIs (Swagger/OpenAPI) 📄")
    st.markdown("""
        Cole o conteúdo de um arquivo OpenAPI (JSON ou YAML) para analisar a especificação da API em busca de:
        - **Vulnerabilidades OWASP API Security Top 10 (2023)**
        - Falhas de design e implementação
        - Exposição de informações sensíveis
        - Boas práticas de segurança e sugestões de melhoria.
    """)

    # Inicializa ou reseta o estado
    if 'swagger_input_content' not in st.session_state:
        st.session_state.swagger_input_content = ""
    if 'swagger_analysis_result' not in st.session_state:
        st.session_state.swagger_analysis_result = [] # Armazena uma lista de objetos (dicionários) se necessário
    if 'swagger_analysis_result_display' not in st.session_state:
        st.session_state.swagger_analysis_result_display = "" # Resultado processado para display
    if 'swagger_context_input' not in st.session_state:
        st.session_state.swagger_context_input = ""
    if 'swagger_summary' not in st.session_state:
        st.session_state.swagger_summary = None

    def reset_swagger_analyzer():
        st.session_state.swagger_input_content = ""
        st.session_state.swagger_analysis_result = []
        st.session_state.swagger_analysis_result_display = ""
        st.session_state.swagger_context_input = ""
        st.session_state.swagger_summary = None
        st.rerun()

    if st.button("Limpar Análise OpenAPI", key="reset_swagger_analysis_button"):
        reset_swagger_analyzer()

    # Campo para colar o conteúdo OpenAPI
    swagger_content = st.text_area(
        "Cole o conteúdo do arquivo OpenAPI (JSON ou YAML) aqui:",
        value=st.session_state.swagger_input_content,
        placeholder="Ex: { 'openapi': '3.0.0', 'info': { ... }, 'paths': { ... } }",
        height=400,
        key="swagger_input_area"
    )
    st.session_state.swagger_input_content = swagger_content.strip()

    context_input = st.text_area(
        "Forneça um contexto adicional sobre a API (opcional):",
        value=st.session_state.swagger_context_input,
        placeholder="Ex: 'Esta API é para gerenciamento de usuários', 'É uma API interna para microserviços'",
        key="swagger_context_input_area"
    )
    st.session_state.swagger_context_input = context_input.strip()

    if st.button("Analisar OpenAPI", key="analyze_swagger_button"):
        if not st.session_state.swagger_input_content:
            st.error("Por favor, cole o conteúdo OpenAPI/Swagger para análise.")
        else:
            with st.spinner("Analisando especificação OpenAPI/Swagger..."):
                # Tenta detectar se é JSON ou YAML para melhor formatação no prompt
                try:
                    json.loads(st.session_state.swagger_input_content)
                    content_format = "JSON"
                    code_lang = "json"
                except json.JSONDecodeError:
                    try:
                        yaml.safe_load(st.session_state.swagger_input_content)
                        content_format = "YAML"
                        code_lang = "yaml"
                    except yaml.YAMLError:
                        content_format = "TEXTO SIMPLES (formato inválido, análise pode ser limitada)"
                        code_lang = "plaintext"
                        st.warning("O conteúdo colado não parece ser um JSON ou YAML válido. A análise pode ser limitada.")

                swagger_prompt = (
                    f"Você é um especialista em segurança de APIs e pentest, com profundo conhecimento na OWASP API Security Top 10 (2023).\n"
                    f"Sua tarefa é analisar a especificação OpenAPI (Swagger) fornecida ({content_format}) e o contexto adicional, identificando **TODAS as possíveis vulnerabilidades de segurança e falhas de design**.\n"
                    f"\n**RESUMO:** Forneça um resumo quantitativo na PRIMEIRA LINHA da sua resposta, no formato exato: `Total de Vulnerabilidades API: X | Críticas: Y | Altas: Z | Médias: W | Baixas: V` (substitua X,Y,Z,W,V pelos números correspondentes). Se não houver vulnerabilidades, use 0.\n\n"
                    f"Para cada **vulnerabilidade ou falha de design** identificada, apresente de forma concisa e prática, utilizando formato Markdown para títulos e blocos de código:\n\n"
                    f"## [Nome da Vulnerabilidade/Falha de Design]\n"
                    f"**Categoria OWASP API Security Top 10 (2023):** [Ex: API1: Broken Object Level Authorization (BOLA), API8: Security Misconfiguration]. Se não se encaixar diretamente, use 'Falha de Design Geral'.\n"
                    f"**Severidade/Risco:** [Crítica/Alta/Média/Baixa - explique o impacto específico para esta API]\n"
                    f"**Localização na Especificação:** Indique o caminho exato ou uma descrição clara de onde a falha foi observada na especificação OpenAPI (ex: `/paths/{userId}/details GET`, `components/schemas/UserObject`).\n"
                    f"**Detalhes e Explicação:** Explique brevemente a falha, como ela se manifesta nesta especificação e o impacto potencial.\n"
                    f"**Exemplo de Cenário de Ataque/PoC (se aplicável):** Descreva um cenário de ataque que explore essa vulnerabilidade, ou um exemplo de requisição HTTP (com `curl` ou similar) que demonstre o problema. Encapsule em um bloco de código Markdown com linguagem `http` ou `bash` (` ```http `, ` ```bash `).\n"
                    f"**Ferramentas Sugeridas:** Liste ferramentas que podem ser usadas para testar ou validar este achado (ex: Postman, Burp Suite, OWASP ZAP, Kiterunner, FFUF, OpenAPI-fuzzer, Dastardly, etc.).\n"
                    f"**Recomendação/Mitigação:** Ações concretas e específicas para corrigir a vulnerabilidade ou melhorar o design da API, relevantes para a especificação OpenAPI fornecida (ex: adicionar autenticação/autorização, aplicar validação de esquema, limitar taxas).\n\n"
                    f"**Conteúdo da Especificação OpenAPI/Swagger (para sua referência):\n"
                    f"```" + code_lang + f"\n{st.session_state.swagger_input_content}\n```\n\n"
                    f"**Contexto Adicional:** {st.session_state.swagger_context_input if st.session_state.swagger_context_input else 'Nenhum contexto adicional fornecido.'}\n\n"
                    f"Se não encontrar vulnerabilidades ou falhas de design óbvias, indique isso claramente e sugira melhorias gerais de segurança para a API.\n"
                    f"Sua resposta deve ser direta, útil e focada em ações e informações completas para um pentester ou desenvolvedor."
                )
                
                analysis_result_raw = obter_resposta_llm(llm_model_text, [swagger_prompt])
                
                if analysis_result_raw:
                    st.session_state.swagger_summary, st.session_state.swagger_analysis_result_display = parse_vulnerability_summary(analysis_result_raw)
                else:
                    st.session_state.swagger_analysis_result_display = "Não foi possível obter a análise da especificação OpenAPI. Tente novamente."
                    st.session_state.swagger_summary = None

    if st.session_state.swagger_analysis_result_display:
        st.subheader("Resultados da Análise OpenAPI")
        if st.session_state.swagger_summary:
            st.markdown("#### Resumo das Vulnerabilidades API Identificadas:")
            cols = st.columns(5)
            cols[0].metric("Total", st.session_state.swagger_summary["Total"])
            cols[1].metric("Críticas", st.session_state.swagger_summary["Críticas"])
            cols[2].metric("Altas", st.session_state.swagger_summary["Altas"])
            cols[3].metric("Médias", st.session_state.swagger_summary["Médias"])
            cols[4].metric("Baixas", st.session_state.swagger_summary["Baixas"])
            st.markdown("---")
        st.markdown(st.session_state.swagger_analysis_result_display)


def search_exploit_page(llm_model_vision, llm_model_text):
    st.header("Search Exploit (NVD): Busca por CVEs e PoCs 🔍") # Título mais claro
    st.markdown("""
        Digite o nome de um software/serviço e sua versão. O SentinelAI irá pesquisar
        por CVEs (Common Vulnerabilities and Exposures) e possíveis Provas de Conceito (PoCs),
        consultando a base de dados oficial da NVD (National Vulnerability Database).
    """)

    st.info("ℹ️ **Informação:** Esta ferramenta consulta a API oficial da NVD (National Vulnerability Database) para obter as CVEs mais recentes. As informações de PoC e ferramentas são geradas pelo modelo de IA com base nas CVEs encontradas e em seu conhecimento de segurança.")
    st.warning("⚠️ **Atenção sobre o erro de conexão:** Se você estiver recebendo o erro 'No connection adapters were found', verifique sua conexão com a internet e se há algum proxy ou configuração de certificado SSL/TLS que possa estar impedindo as requisições Python. A NVD API requer acesso HTTPS padrão.")


    # Inicializa ou reseta o estado
    if 'exploit_software_name' not in st.session_state:
        st.session_state.exploit_software_name = ""
    if 'exploit_software_version' not in st.session_state:
        st.session_state.exploit_software_version = ""
    if 'exploit_analysis_result' not in st.session_state:
        st.session_state.exploit_analysis_result = ""
    if 'exploit_summary' not in st.session_state:
        st.session_state.exploit_summary = None
    if 'nvd_search_results' not in st.session_state:
        st.session_state.nvd_search_results = []

    def reset_exploit_search():
        st.session_state.exploit_software_name = ""
        st.session_state.exploit_software_version = ""
        st.session_state.exploit_analysis_result = ""
        st.session_state.exploit_summary = None
        st.session_state.nvd_search_results = []
        st.rerun()

    if st.button("Limpar Busca", key="reset_exploit_search_button"):
        reset_exploit_search()

    # Campos de entrada
    software_name = st.text_input(
        "Nome do Software/Serviço (Ex: Apache HTTP Server, MySQL, OpenSSL):",
        value=st.session_state.exploit_software_name,
        placeholder="Ex: Apache HTTP Server",
        key="exploit_software_name_input"
    )
    st.session_state.exploit_software_name = software_name.strip()

    software_version = st.text_input(
        "Versão (Opcional, para refinar a busca. Ex: 2.4.50, 8.0.30):",
        value=st.session_state.exploit_software_version,
        placeholder="Ex: 2.4.50 ou 8.0.30",
        key="exploit_software_version_input"
    )
    st.session_state.exploit_software_version = software_version.strip()

    # Define version_text aqui, para que esteja sempre disponível
    version_text = f" versão '{st.session_state.exploit_software_version}'" if st.session_state.exploit_software_version else ""


    if st.button("Buscar CVEs na NVD", key="search_exploit_button"): # Alterado o texto do botão
        if not st.session_state.exploit_software_name:
            st.error("Por favor, digite o nome do software/serviço para buscar.")
            st.session_state.nvd_search_results = []
        else:
            st.session_state.nvd_search_results = []
            st.session_state.exploit_analysis_result = ""
            st.session_state.exploit_summary = None

            # --- 1. Consulta à API da NVD ---
            with st.spinner(f"Consultando NVD para {st.session_state.exploit_software_name} {version_text}..."):
                base_nvd_url = "[https://services.nvd.nist.gov/rest/json/cves/2.0](https://services.nvd.nist.gov/rest/json/cves/2.0)"
                
                params = {}
                search_term = st.session_state.exploit_software_name
                if st.session_state.exploit_software_version:
                    params["keywordSearch"] = f"{st.session_state.exploit_software_name} {st.session_state.exploit_software_version}"
                else:
                    params["keywordSearch"] = st.session_state.exploit_software_name
                
                params["resultsPerPage"] = 10 
                
                try:
                    headers = {}
                    if NVD_API_KEY: # Verifica se a chave existe antes de adicionar
                        headers["apiKey"] = NVD_API_KEY # A NVD usa o cabeçalho 'apiKey'
                        st.info("Usando NVD API Key para a consulta.") # Opcional: para feedback

                    response = requests.get(base_nvd_url, params=params, headers=headers, timeout=15)
                    response.raise_for_status() # Lança um erro para status HTTP ruins (4xx ou 5xx)
                    nvd_data = response.json()
                    if nvd_data and 'vulnerabilities' in nvd_data:
                        st.session_state.nvd_search_results = nvd_data['vulnerabilities']
                        if st.session_state.nvd_search_results:
                            st.success(f"Encontradas {len(st.session_state.nvd_search_results)} vulnerabilidades na NVD.")
                        else:
                            st.info(f"Nenhuma vulnerabilidade específica encontrada na NVD para '{search_term}'. O LLM tentará fornecer informações gerais.")
                    else:
                        st.info(f"Nenhum resultado de vulnerabilidade retornado pela NVD para '{search_term}'. O LLM tentará fornecer informações gerais.")

                except requests.exceptions.RequestException as e:
                    # Mensagem de erro mais focada na conexão
                    st.error(f"Erro ao consultar a API da NVD: {e}. Verifique sua conexão com a internet ou possíveis proxies/firewalls. O LLM tentará fornecer informações gerais com base em seu conhecimento.")
                    st.session_state.nvd_search_results = []
                except json.JSONDecodeError:
                    st.error(f"Erro ao decodificar a resposta JSON da NVD. Pode ser um problema temporário da API. O LLM tentará fornecer informações gerais com base em seu conhecimento.")
                    st.session_state.nvd_search_results = []

            # --- 2. Alimentar o LLM com os resultados da NVD ---
            full_llm_context = ""
            if st.session_state.nvd_search_results:
                nvd_results_for_llm = []
                for vuln in st.session_state.nvd_search_results:
                    cve_id = vuln['cve']['id']
                    description = "N/A"
                    if 'descriptions' in vuln['cve'] and vuln['cve']['descriptions']:
                        for desc_entry in vuln['cve']['descriptions']: # Renomeado para evitar conflito
                            if desc_entry['lang'] == 'en': # Preferir descrição em inglês
                                description = desc_entry['value']
                                break
                    severity = "Não especificado"
                    # Tenta obter a severidade CVSSv3.1 primeiro, depois v2.0
                    if 'metrics' in vuln['cve']:
                        if 'cvssMetricV31' in vuln['cve']['metrics'] and vuln['cve']['metrics']['cvssMetricV31']:
                            for metric in vuln['cve']['metrics']['cvssMetricV31']:
                                if 'cvssData' in metric and 'baseSeverity' in metric['cvssData']:
                                    severity = metric['cvssData']['baseSeverity'] + " (CVSS v3.1)"
                                    break
                        elif 'cvssMetricV2' in vuln['cve']['metrics'] and vuln['cve']['metrics']['cvssMetricV2']:
                            for metric in vuln['cve']['metrics']['cvssMetricV2']:
                                if 'cvssData' in metric and 'baseSeverity' in metric['cvssData']:
                                    severity = metric['cvssData']['baseSeverity'] + " (CVSS v2.0)"
                                    break

                    nvd_results_for_llm.append(f"CVE ID: {cve_id}\nDescrição: {description}\nSeveridade NVD: {severity}\n")
                
                full_llm_context = "\n".join(nvd_results_for_llm) + "\n\n"
                llm_instruction_prefix = f"A NVD retornou as seguintes vulnerabilidades para '{st.session_state.exploit_software_name}' {version_text}:\n\n```\n{full_llm_context}\n```\n\nCom base nessas informações da NVD e em seu conhecimento de segurança, para cada CVE, detalhe a vulnerabilidade e forneça informações sobre possíveis PoCs e ferramentas. Foque em:\n"
            else:
                llm_instruction_prefix = f"Não foram encontrados resultados específicos na NVD para '{st.session_state.exploit_software_name}' {version_text}. Com base em seu conhecimento geral sobre este software/serviço, forneça informações sobre vulnerabilidades comuns, técnicas de exploração e ferramentas relevantes. Foque em:\n"

            
            exploit_prompt = (
                f"Você é um especialista em segurança de sistemas e pentest, com amplo conhecimento em bancos de dados de vulnerabilidades como Exploit-DB, NVD e metasploit. Sua tarefa é analisar as informações fornecidas e gerar um relatório detalhado de vulnerabilidades.\n"
                f"{llm_instruction_prefix}"
                f"\n**RESUMO:** Forneça um resumo quantitativo na PRIMEIRA LINHA da sua resposta, no formato exato: `Total de Vulnerabilidades: X | Críticas: Y | Altas: Z | Médias: W | Baixas: V` (substitua X,Y,Z,W,V pelos números correspondentes). Se não houver vulnerabilidades, use 0.\n\n"
                f"Para cada vulnerabilidade/exploit relevante, apresente de forma concisa e prática, utilizando formato Markdown:\n\n"
                f"## [Nome do Exploit/Vulnerabilidade] (Ex: Apache Struts2 Remote Code Execution)\n"
                f"**CVE ID:** [ID da CVE, Ex: CVE-2023-12345] / **Severidade NVD:** [Severidade da NVD, se disponível]\n" # Foco na NVD
                f"**Software/Serviço Afetado:** {st.session_state.exploit_software_name} {version_text}\n"
                f"**Tipo:** [Ex: RCE, LFI, Escalation de Privilégios, DoS, SQLi, etc.]\n"
                f"**Severidade Avaliada:** [Crítica/Alta/Média/Baixa - Avalie a severidade geral com base nas informações da NVD e seu conhecimento de pentest]\n"
                f"**Descrição da CVE:** Explique a vulnerabilidade e seu impacto, sintetizando a informação da NVD. Mencione as condições específicas para exploração.\n"
                f"**PoC (Prova de Conceito) / Método de Exploração:** Forneça um exemplo prático de como esta vulnerabilidade *poderia ser explorada*. Inclua comandos de terminal, trechos de código (Python, Ruby, C, etc.), payloads ou requisições HTTP (`curl`). Adapte o PoC para o software/serviço especificado. **Encapsule o código/comando em um bloco de código Markdown (` ``` ` com a linguagem apropriada).**\n"
                f"**Ferramentas Sugeridas:** Liste ferramentas (Ex: Metasploit, Nmap Scripting Engine (NSE), Nessus, OpenVAS, custom scripts Python/Perl/Ruby) que podem ser usadas para detectar ou explorar a vulnerabilidade.\n"
                f"**Mitigação/Solução:** Ações recomendadas para corrigir ou mitigar a vulnerabilidade.\n\n"
                f"Se, mesmo após a consulta, não houver exploits ou CVEs diretos, ou se o modelo não tiver conhecimento específico, indique isso claramente e sugira métodos genéricos de pentest para esse tipo de software/serviço ou explique que o conhecimento é limitado à sua data de treinamento para detalhes não contidos nas fontes fornecidas.\n"
                f"Seu objetivo é fornecer informações acionáveis para um pentester."
            )

            exploit_analysis_raw = obter_resposta_llm(llm_model_text, [exploit_prompt])
            
            if exploit_analysis_raw:
                st.session_state.exploit_summary, st.session_state.exploit_analysis_result = parse_vulnerability_summary(exploit_analysis_raw)
            else:
                st.session_state.exploit_analysis_result = "Não foi possível obter uma resposta do Gemini. Tente novamente."
                st.session_state.exploit_summary = None

    if st.session_state.exploit_analysis_result:
        st.subheader("Resultados da Busca de Exploit/CVEs")
        if st.session_state.exploit_summary:
            st.markdown("#### Resumo das Vulnerabilidades Identificadas:")
            cols = st.columns(5)
            cols[0].metric("Total", st.session_state.exploit_summary["Total"])
            cols[1].metric("Críticas", st.session_state.exploit_summary["Críticas"])
            cols[2].metric("Altas", st.session_state.exploit_summary["Altas"])
            cols[3].metric("Médias", st.session_state.exploit_summary["Médias"])
            cols[4].metric("Baixas", st.session_state.exploit_summary["Baixas"])
            st.markdown("---")
        st.markdown(st.session_state.exploit_analysis_result)


# --- Novo Módulo: Acunetix Scanner Insights ---

def acunetix_insights_page(llm_model_vision, llm_model_text):
    st.header("Acunetix Scanner Insights 🕷️")
    st.markdown("""
        Cole o output de um scan do Acunetix (preferencialmente em formato JSON ou XML para melhor precisão)
        ou forneça o ID de um scan existente (se o Acunetix for acessível via API).
        O SentinelAI irá analisar as vulnerabilidades encontradas pelo Acunetix, fornecer insights,
        mapear para OWASP (se aplicável), e sugerir PoCs ou próximas etapas para validação manual.
    """)

    if not ACUNETIX_API_KEY or not ACUNETIX_URL:
        st.warning("⚠️ **Configuração Faltando:** Para usar a integração direta com a API do Acunetix, configure `ACUNETIX_API_KEY` e `ACUNETIX_URL` no seu arquivo `.env`.")
        st.info("Você pode obter sua API Key do Acunetix em 'Profile > API Key' e a URL é a base da sua instalação Acunetix (ex: `https://myacunetix.com`).")

    # Inicializa ou reseta o estado
    if 'acunetix_input_type' not in st.session_state:
        st.session_state.acunetix_input_type = "paste_output"
    if 'acunetix_pasted_output' not in st.session_state:
        st.session_state.acunetix_pasted_output = ""
    if 'acunetix_scan_id' not in st.session_state:
        st.session_state.acunetix_scan_id = ""
    if 'acunetix_analysis_result' not in st.session_state:
        st.session_state.acunetix_analysis_result = ""
    if 'acunetix_summary' not in st.session_state:
        st.session_state.acunetix_summary = None
    if 'acunetix_fetch_error' not in st.session_state:
        st.session_state.acunetix_fetch_error = False

    def reset_acunetix_analysis():
        st.session_state.acunetix_input_type = "paste_output"
        st.session_state.acunetix_pasted_output = ""
        st.session_state.acunetix_scan_id = ""
        st.session_state.acunetix_analysis_result = ""
        st.session_state.acunetix_summary = None
        st.session_state.acunetix_fetch_error = False
        st.rerun()

    if st.button("Limpar Análise Acunetix", key="reset_acunetix_analysis_button"):
        reset_acunetix_analysis()

    # Seleção do tipo de entrada
    input_type = st.radio(
        "Como você deseja fornecer os dados do Acunetix?",
        ["Colar Output do Scan", "Buscar Scan por ID (via API Acunetix)"],
        key="acunetix_input_type_radio",
        index=0 if st.session_state.acunetix_input_type == "paste_output" else 1
    )
    st.session_state.acunetix_input_type = "paste_output" if input_type == "Colar Output do Scan" else "fetch_by_id"

    scan_data_to_analyze = ""
    
    if st.session_state.acunetix_input_type == "paste_output":
        pasted_output = st.text_area(
            "Cole o output completo do scan do Acunetix (JSON ou XML):",
            value=st.session_state.acunetix_pasted_output,
            height=400,
            placeholder="Ex: <scan-results>...</scan-results> ou { 'vulnerabilities': [...] }",
            key="acunetix_pasted_output_area"
        )
        st.session_state.acunetix_pasted_output = pasted_output.strip()
        scan_data_to_analyze = st.session_state.acunetix_pasted_output

    elif st.session_state.acunetix_input_type == "fetch_by_id":
        scan_id = st.text_input(
            "ID do Scan no Acunetix:",
            value=st.session_state.acunetix_scan_id,
            placeholder="Ex: 123e4567-e89b-12d3-a456-426614174000",
            key="acunetix_scan_id_input"
        )
        st.session_state.acunetix_scan_id = scan_id.strip()

        if st.button("Buscar Scan na API Acunetix", key="fetch_acunetix_scan_button"):
            if not ACUNETIX_API_KEY or not ACUNETIX_URL:
                st.error("Por favor, configure `ACUNETIX_API_KEY` e `ACUNETIX_URL` no seu arquivo `.env` para usar esta funcionalidade.")
                st.session_state.acunetix_fetch_error = True
            elif not st.session_state.acunetix_scan_id:
                st.error("Por favor, digite o ID do Scan.")
                st.session_state.acunetix_fetch_error = True
            else:
                with st.spinner(f"Buscando detalhes do scan ID {st.session_state.acunetix_scan_id} na API do Acunetix..."):
                    # A API do Acunetix para resultados de scan é tipicamente assim:
                    # GET /api/v1/scans/{scan_id}/vulnerabilities ou /api/v1/scans/{scan_id}/results
                    # A URL exata e os cabeçalhos podem variar ligeiramente com a versão do Acunetix.
                    
                    api_endpoint = f"{ACUNETIX_URL.rstrip('/')}/api/v1/scans/{st.session_state.acunetix_scan_id}/results" # Exemplo de endpoint
                    headers = {
                        "X-Auth": ACUNETIX_API_KEY,
                        "Content-Type": "application/json"
                    }
                    
                    try:
                        response = requests.get(api_endpoint, headers=headers, timeout=30)
                        response.raise_for_status() # Lança um erro para status HTTP ruins
                        scan_data_from_api = response.json()
                        st.session_state.acunetix_pasted_output = json.dumps(scan_data_from_api, indent=2) # Armazena para visualização
                        st.success(f"Dados do scan {st.session_state.acunetix_scan_id} obtidos da API.")
                        st.session_state.acunetix_fetch_error = False
                    except requests.exceptions.RequestException as e:
                        st.error(f"Erro ao buscar scan na API do Acunetix: {e}. Verifique a URL, API Key e ID do Scan.")
                        st.session_state.acunetix_fetch_error = True
                        st.session_state.acunetix_pasted_output = "" # Limpa qualquer dado anterior
                    except json.JSONDecodeError:
                        st.error("Erro ao decodificar a resposta JSON da API do Acunetix.")
                        st.session_state.acunetix_fetch_error = True
                        st.session_state.acunetix_pasted_output = ""
        
        if st.session_state.acunetix_pasted_output:
            st.subheader("Output do Scan do Acunetix (Obtido via API):")
            st.code(st.session_state.acunetix_pasted_output, language="json") # Supondo JSON

        scan_data_to_analyze = st.session_state.acunetix_pasted_output # Usa o output recuperado para análise

    if st.button("Analisar Output do Acunetix com LLM", key="analyze_acunetix_button"):
        if not scan_data_to_analyze:
            st.error("Por favor, cole o output do scan ou busque um scan via ID primeiro.")
        else:
            with st.spinner("Analisando vulnerabilidades do Acunetix com LLM..."):
                # Tenta parsear o input para JSON ou XML
                parsed_scan_data = None
                input_format = "texto"
                try:
                    parsed_scan_data = json.loads(scan_data_to_analyze)
                    input_format = "JSON"
                except json.JSONDecodeError:
                    try:
                        # Para XML, pode ser necessário um parser mais robusto como ElementTree
                        # Aqui, apenas verificamos se parece XML para informar ao LLM
                        if scan_data_to_analyze.strip().startswith('<'):
                            input_format = "XML"
                    except Exception:
                        pass # Continua como texto se não for nem JSON nem XML aparente

                prompt_acunetix = (
                    f"Você é um especialista em segurança da informação e pentest, com experiência na análise de relatórios de scanners de vulnerabilidade como o Acunetix. Sua tarefa é analisar o output de um scan do Acunetix fornecido (em formato {input_format}) e extrair as vulnerabilidades mais importantes, fornecendo insights práticos para um pentester.\n"
                    f"\n**Output do Scan do Acunetix ({input_format}):**\n```\n{scan_data_to_analyze}\n```\n\n"
                    f"**RESUMO:** Forneça um resumo quantitativo na PRIMEIRA LINHA da sua resposta, no formato exato: `Total de Vulnerabilidades: X | Críticas: Y | Altas: Z | Médias: W | Baixas: V` (substitua X,Y,Z,W,V pelos números correspondentes). Se não houver vulnerabilidades, use 0.\n\n"
                    f"Para cada **vulnerabilidade importante** identificada pelo Acunetix, forneça os seguintes detalhes de forma concisa e prática, utilizando formato Markdown:\n\n"
                    f"## [Nome da Vulnerabilidade] (Ex: SQL Injection, Cross-Site Scripting)\n"
                    f"**Severidade Acunetix:** [Severidade reportada pelo Acunetix, ex: High, Medium]\n"
                    f"**Localização/URL:** https://www.reddit.com/r/AfterEffects/comments/3taw58/cant_export_project/\n"
                    f"**Categoria OWASP (2021) / CWE (se aplicável):** [Mapeie para a categoria OWASP Top 10 mais relevante, ex: A03: Injection. Se houver CWE no output, inclua também.]\n"
                    f"**Descrição e Implicações:** Explique brevemente a vulnerabilidade, como ela foi detectada pelo Acunetix (se o output detalhar) e qual o impacto potencial.\n"
                    f"**PoC (Prova de Conceito) / Como Reproduzir:** Com base nas informações do scan e em seu conhecimento, descreva os passos para reproduzir manualmente a vulnerabilidade. Se o Acunetix fornecer o PoC, explique-o de forma clara e forneça um exemplo adaptado. **Encapsule exemplos de payloads/requisições em blocos de código Markdown (` ``` ` com a linguagem apropriada).**\n"
                    f"**Ferramentas Sugeridas para Validação Manual:** Liste ferramentas (Ex: Burp Suite, OWASP ZAP, Postman, curl, ferramentas específicas para a vulnerabilidade) que um pentester usaria para validar ou explorar mais a fundo essa vulnerabilidade.\n"
                    f"**Recomendação/Mitigação:** Ações concretas e específicas para corrigir a vulnerabilidade.\n\n"
                    f"Se o output não contiver vulnerabilidades, ou se for muito genérico, indique isso claramente. Priorize as vulnerabilidades de maior severidade.\n"
                    f"Seu objetivo é transformar o relatório do Acunetix em insights acionáveis para um pentester."
                )

                acunetix_analysis_raw = obter_resposta_llm(llm_model_text, [prompt_acunetix])
                
                if acunetix_analysis_raw:
                    st.session_state.acunetix_summary, st.session_state.acunetix_analysis_result = parse_vulnerability_summary(acunetix_analysis_raw)
                else:
                    st.session_state.acunetix_analysis_result = "Não foi possível analisar o output do Acunetix. Tente novamente ou forneça um formato mais claro."
                    st.session_state.acunetix_summary = None

    if st.session_state.acunetix_analysis_result:
        st.subheader("Resultados da Análise do Acunetix")
        if st.session_state.acunetix_summary:
            st.markdown("#### Resumo das Vulnerabilidades Identificadas:")
            cols = st.columns(5)
            cols[0].metric("Total", st.session_state.acunetix_summary["Total"])
            cols[1].metric("Críticas", st.session_state.acunetix_summary["Críticas"])
            cols[2].metric("Altas", st.session_state.acunetix_summary["Altas"])
            cols[3].metric("Médias", st.session_state.acunetix_summary["Médias"])
            cols[4].metric("Baixas", st.session_state.acunetix_summary["Baixas"])
            st.markdown("---")
        st.markdown(st.session_state.acunetix_analysis_result)

# --- Lógica Principal do Aplicativo ---

# Garante que os modelos LLM são inicializados
# Todas as variáveis de session_state são inicializadas aqui para evitar AttributeError
if 'llm_models_initialized' not in st.session_state:
    st.session_state.llm_models_initialized = False # Será True após a primeira inicialização
    st.session_state.llm_vision_model = None
    st.session_state.llm_text_model = None
    # Variáveis para Análise por Texto (OWASP)
    st.session_state.owasp_text_input_falha = ""
    st.session_state.owasp_text_analysis_result = ""
    st.session_state.owasp_text_context_input = ""
    st.session_state.owasp_text_consider_waf_state = False
    # Variáveis para Análise de Requisições HTTP
    st.session_state.http_request_input_url = ""
    st.session_state.http_request_input_raw = ""
    st.session_state.http_request_analysis_result = ""
    st.session_state.http_request_consider_waf_state = False
    st.session_state.http_request_summary = None
    # Variáveis para OWASP Image Analyzer
    st.session_state.owasp_image_uploaded = None
    st.session_state.owasp_question_text = ""
    st.session_state.owasp_analysis_result = ""
    st.session_state.owasp_consider_waf_state = False
    # Variáveis para Modelagem de Ameaças (STRIDE)
    st.session_state.stride_image_uploaded = None
    st.session_state.stride_description_text = ""
    st.session_state.stride_analysis_result = ""
    st.session_state.stride_summary = None
    # Variáveis para Pentest Lab
    st.session_state.lab_vulnerability_selected = None
    st.session_state.lab_html_poc = ""
    st.session_state.lab_explanation = ""
    st.session_state.lab_payload_example = ""
    # Variáveis para PoC Generator (HTML)
    st.session_state.poc_gen_vulnerability_input = ""
    st.session_state.poc_gen_context_input = ""
    st.session_state.poc_gen_html_output = ""
    st.session_state.poc_gen_instructions = ""
    st.session_state.poc_gen_payload_example = ""
    # Variáveis para Swagger/OpenAPI Analyzer
    st.session_state.swagger_input_content = ""
    st.session_state.swagger_analysis_result = [] # Agora armazena uma LISTA de objetos (dicionários)
    st.session_state.swagger_analysis_result_display = "" # Resultado processado para display
    st.session_state.swagger_context_input = ""
    st.session_state.swagger_summary = None
    # Variáveis para Static Code Analyzer (Basic)
    st.session_state.code_input_content = ""
    st.session_state.code_analysis_result = ""
    st.session_state.code_language_selected = "Python" # Default
    # Variáveis para Search Exploit (NVD APENAS)
    st.session_state.exploit_software_name = ""
    st.session_state.exploit_software_version = ""
    st.session_state.exploit_analysis_result = ""
    st.session_state.exploit_summary = None
    st.session_state.nvd_search_results = []
    # Variáveis para Acunetix Scanner Insights (NOVO)
    st.session_state.acunetix_input_type = "paste_output"
    st.session_state.acunetix_pasted_output = ""
    st.session_state.acunetix_scan_id = ""
    st.session_state.acunetix_analysis_result = ""
    st.session_state.acunetix_summary = None
    st.session_state.acunetix_fetch_error = False


    # Inicializa os modelos LLM apenas uma vez
    st.session_state.llm_vision_model, st.session_state.llm_text_model = get_gemini_models()
    st.session_state.llm_models_initialized = True
else:
    llm_model_vision = st.session_state.llm_vision_model
    llm_model_text = st.session_state.llm_text_model


# Define as opções de menu na barra lateral
selected_page = st.sidebar.radio(
    "Navegação",
    ["Início", "OWASP Vulnerability Details", "Análise de Requisições HTTP", "OWASP Image Analyzer", "Modelagem de Ameaças (STRIDE)", "Pentest Lab", "PoC Generator (HTML)", "OpenAPI Analyzer", "Static Code Analyzer", "Search Exploit (NVD)", "Acunetix Scanner Insights"], # Adicionada nova página
    index=0 # Página inicial padrão "Início"
)

# Renderiza a página selecionada baseando-se na escolha do usuário
if selected_page == "Início":
    home_page()
elif selected_page == "OWASP Vulnerability Details":
    owasp_text_analysis_page(llm_model_vision, llm_model_text)
elif selected_page == "Análise de Requisições HTTP":
    http_request_analysis_page(llm_model_vision, llm_model_text)
elif selected_page == "OWASP Image Analyzer":
    owasp_scout_visual_page(llm_model_vision, llm_model_text)
elif selected_page == "Modelagem de Ameaças (STRIDE)":
    modelagem_de_ameacas_page(llm_model_vision, llm_model_text)
elif selected_page == "Pentest Lab":
    pentest_lab_page(llm_model_vision, llm_model_text)
elif selected_page == "PoC Generator (HTML)":
    poc_generator_html_page(llm_model_vision, llm_model_text)
elif selected_page == "OpenAPI Analyzer":
    swagger_openapi_analyzer_page(llm_model_vision, llm_model_text)
elif selected_page == "Static Code Analyzer":
    static_code_analyzer_page(llm_model_vision, llm_model_text)
elif selected_page == "Search Exploit (NVD)":
    search_exploit_page(llm_model_vision, llm_model_text)
elif selected_page == "Acunetix Scanner Insights": # Nova página adicionada
    acunetix_insights_page(llm_model_vision, llm_model_text)
