import os
import sys
import warnings
import time
import random
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

# === Carrega variáveis de ambiente ===
load_dotenv()

# === Suprime avisos de depreciação desnecessários ===
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

class Nutureo:
    def __init__(self, session_id: str, db_path: str = 'sqlite:///memory.db'):
        # ✅ Verificação rigorosa da chave da API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("❌ A variável de ambiente 'GOOGLE_API_KEY' não foi definida ou está vazia.")

        try:
            # 🤖 Inicializa o modelo Gemini com tolerância a falhas futura
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.1,
                max_output_tokens=2048,  # Ajustável futuramente
            )

            # 💬 Histórico persistente da conversa
            self.chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=db_path
            )

            self.memory = ConversationBufferMemory(
                memory_key='chat_history',
                chat_memory=self.chat_history,
                return_messages=True
            )

            # ⚙️ Ferramentas que o agente pode usar
            self.tools = [
                Tool(
                    name="nutritional_advice",
                    func=self.provide_nutritional_advice,
                    description="Fornece conselhos nutricionais com base em evidências científicas."
                )
            ]

            # 🧾 Prompt de sistema com personalidade do Nutureo
            system_prompt = '''
Nome do Agente: Nutureo, o Mestre Nutrólogo Supremo

Você fala em PT-BR sempre, exceto se te pedirem outro idioma.

Contexto:
Nutureo é uma IA de ponta no campo da nutrição, com expertise em bioquímica e dietas como mediterrânea, cetogênica e ayurvédica. Consultado por celebridades, ele cria planos alimentares personalizados com base em ciência e cultura alimentar global.

Personalidade:
- ENTP: Extrovertido, criativo, lógico
- Comunicação clara, científica e com leve humor
- Explica conceitos complexos de forma simples

Objetivo:
Empoderar o usuário a tomar decisões alimentares saudáveis, com base em ciência e propósito.
            '''

            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                agent_kwargs={"system_prompt": system_prompt},
                verbose=False
            )

        except Exception as e:
            print("🚨 Falha ao inicializar o agente Nutureo:", e)
            sys.exit(1)

    def provide_nutritional_advice(self, query: str) -> str:
        try:
            # 💡 Lógica de conselho pode ser aprimorada com base em dados reais futuramente
            return f"🍏 Conselho nutricional baseado em sua dúvida: '{query}'"
        except Exception as e:
            print("⚠️ Erro ao fornecer conselho nutricional:", e)
            return "⚠️ Ocorreu um problema ao gerar o conselho."

    def run_with_retry(self, input_text: str, max_retries: int = 3) -> str:
        retries = 0
        while retries < max_retries:
            try:
                result = self.agent.invoke({"input": input_text})
                return result.get("output", "❗ Resposta não encontrada.")
            except Exception as e:
                retries += 1
                print(f"⚠️ Tentativa {retries}/{max_retries} falhou: {e}")
                if retries >= max_retries:
                    return "⚠️ Excedemos o número máximo de tentativas. Tente novamente mais tarde."
                else:
                    # Delay progressivo para retry
                    wait_time = random.randint(2, 5) ** retries
                    print(f"⏳ Tentando novamente em {wait_time} segundos...")
                    time.sleep(wait_time)
        return "⚠️ Algo deu errado durante o processo, tente mais tarde."

    def run(self, input_text: str) -> str:
        return self.run_with_retry(input_text)

def main():
    print("🤖 Nutureo está pronto para ajudá-lo com suas dúvidas nutricionais!")
    print("Digite 'sair' para encerrar a conversa.\n")

    session_id = "sessao_001"

    try:
        nutureo = Nutureo(session_id=session_id)
    except EnvironmentError as e:
        print(e)
        return

    while True:
        try:
            user_input = input("Você: ").strip()
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Nutureo: Até a próxima! 🥗 Mantenha uma alimentação saudável.")
                break

            resposta = nutureo.run(user_input)
            print(f"Nutureo: {resposta}\n")

        except KeyboardInterrupt:
            print("\n👋 Encerrando sessão com Nutureo. Até logo!")
            break
        except Exception as e:
            print("💥 Erro inesperado na interface:", e)
            continue

if __name__ == "__main__":
    main()
