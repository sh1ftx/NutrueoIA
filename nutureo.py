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
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("❌ A variável de ambiente 'GOOGLE_API_KEY' não foi definida ou está vazia.")

        try:
            # Inicializa o modelo
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
                max_output_tokens=2048,
            )

            # Histórico persistente
            self.chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=db_path
            )

            self.memory = ConversationBufferMemory(
                memory_key='chat_history',
                chat_memory=self.chat_history,
                return_messages=True
            )

            # Ferramentas disponíveis para o agente
            self.tools = [
                Tool(
                    name="Conselho Nutricional",
                    func=self.provide_nutritional_advice,
                    description="Use esta ferramenta para fornecer conselhos nutricionais personalizados baseados em ciência."
                )
            ]

            # Prompt do sistema
            system_prompt = '''
Você é Nutureo, o Mestre Nutrólogo Supremo. Sempre fale em português, exceto se solicitado outro idioma.

Contexto:
Você é uma IA especialista em nutrição com conhecimento profundo em bioquímica, dietas (mediterrânea, cetogênica, ayurvédica), e comportamento alimentar. Suas recomendações são baseadas em evidências científicas e cultura alimentar.

Personalidade:
- Extrovertido, criativo, lógico (ENTP)
- Comunicação clara e com leve humor
- Capacidade de simplificar conceitos complexos

Objetivo:
Ajudar o usuário a tomar decisões alimentares mais saudáveis e conscientes, com base em ciência e propósito.
            '''

            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                memory=self.memory,
                agent_kwargs={
                    "system_message": system_prompt.strip()
                },
                verbose=False
            )

        except Exception as e:
            print("🚨 Falha ao inicializar o agente Nutureo:", e)
            sys.exit(1)

    def provide_nutritional_advice(self, query: str) -> str:
        try:
            resposta = self.llm.predict(f"Você é um especialista em nutrição. Responda com base científica a esta dúvida: {query}")
            return resposta
        except Exception as e:
            print("⚠️ Erro ao fornecer conselho nutricional:", e)
            return "⚠️ Ocorreu um problema ao gerar o conselho."

    def run_with_retry(self, input_text: str, max_retries: int = 3) -> str:
        retries = 0
        while retries < max_retries:
            try:
                result = self.agent.invoke({"input": input_text})
                return result["output"]  # <-- AQUI: Corrigido para retornar só a resposta do bot
            except Exception as e:
                retries += 1
                print(f"⚠️ Tentativa {retries}/{max_retries} falhou: {e}")
                if retries >= max_retries:
                    return "⚠️ Excedemos o número máximo de tentativas. Tente novamente mais tarde."
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
