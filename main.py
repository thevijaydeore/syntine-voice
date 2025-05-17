import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import our knowledge base
from knowledge_base import KnowledgeBase


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self, knowledge_base=None) -> None:
        # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
        # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
        # Learn more and pick the best one for your app:
        # https://docs.livekit.io/agents/plugins
        super().__init__(
            instructions="You are an AI voice assistant for ElectroMech Pvt. Ltd., a company that offers industrial automation products and solutions such as PLCs, HMIs, VFDs, SCADA systems, and complete automation project execution services. You communicate with customers who may be factory managers, procurement officers, engineers, or business owners from manufacturing industries looking to adopt or upgrade their automation solutions. Your goal is to greet callers professionally, understand their requirements (product inquiries, technical support, service requests, partnership opportunities, or training), answer basic queries related to Tyton’s products and services, collect caller details such as name, company, contact information, and specific needs, and forward complex or custom queries to the human team or take a message with proper context. You should be polite, clear, and concise, use simple, professional language, and guide the caller logically through the conversation without overwhelming them.",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=deepgram.TTS(),
            # use LiveKit's transformer-based turn detector
            turn_detection=MultilingualModel(),
        )
        
        # Store the knowledge base
        self.knowledge_base = knowledge_base
        
    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        self.session.generate_reply(
            instructions="Hey, I'm your ElectroMech product assistant. How can I help you today?", 
            allow_interruptions=True
        )
        
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Perform RAG lookup based on the user's message"""
        # Skip if no knowledge base or no message content
        if not self.knowledge_base or not new_message.text_content:
            return
            
        # Get the user's query
        query = new_message.text_content
        logger.info(f"Performing RAG lookup for query: {query}")
        
        try:
            # Search the knowledge base
            results = self.knowledge_base.search(query, k=2)  # Get top 2 most relevant chunks
            
            if results:
                # Format the context to add to the chat
                context = "\n\n".join(results)
                logger.info(f"Adding {len(results)} chunks to context")
                
                # Add the context to the chat
                turn_ctx.add_message(
                    role="assistant", 
                    content=f"Additional information from knowledge base: {context}"
                )
            else:
                logger.info("No relevant information found in knowledge base")
        except Exception as e:
            logger.error(f"Error in RAG lookup: {e}")
            # Continue without adding context


def prewarm(proc: JobProcess):
    # Load VAD model
    proc.userdata["vad"] = silero.VAD.load()
    
    # Initialize knowledge base
    logger.info("Initializing knowledge base in prewarm function")
    try:
        kb = KnowledgeBase(persist_directory="data/vectorstore")
        
        # Check if we need to create the knowledge base
        if not os.path.exists("data/vectorstore") or not kb.load():
            logger.info("Creating new knowledge base from PDF")
            kb.add_pdf("data/electromech.pdf")
        else:
            logger.info("Loaded existing knowledge base")
            
        # Store the knowledge base in userdata
        proc.userdata["knowledge_base"] = kb
        logger.info("Knowledge base initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}")
        # Continue without knowledge base if there's an error


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    # Get the knowledge base from userdata (if available)
    knowledge_base = ctx.proc.userdata.get("knowledge_base")
    if knowledge_base:
        logger.info("Using knowledge base for the agent")
    else:
        logger.warning("Knowledge base not available, agent will run without RAG")
    
    # Create the assistant with the knowledge base
    assistant = Assistant(knowledge_base=knowledge_base)
    
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            # enable background voice & noise cancellation, powered by Krisp
            # included at no additional cost with LiveKit Cloud
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
