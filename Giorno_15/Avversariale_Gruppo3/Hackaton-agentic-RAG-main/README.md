# MOCA AI Normative Consultant

![MOCA AI](https://img.shields.io/badge/AI-Powered-blue) ![RAG](https://img.shields.io/badge/RAG-Enabled-green) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

> *Un assistente intelligente per navigare la conformità dei Materiali e Oggetti a Contatto con Alimenti.*

## Overview

MOCA AI Normative Consultant is an AI-powered application designed to provide expert guidance on MOCA (Materials and Objects in Contact with Food) regulations. The system leverages Retrieval Augmented Generation (RAG) to deliver accurate, context-aware responses based on authoritative regulatory documents.

The application offers two main interfaces:
- A **Chat Interface** for direct regulatory inquiries
- An **Email Management Interface** for analyzing and responding to customer inquiries

## Key Features

- **Smart Document Retrieval**: Semantic search across regulatory documents
- **Multi-Chat Management**: Create and manage multiple conversations
- **Dynamic Memory**: Override information with custom updates during chat sessions
- **Email Analysis**: Automatically extract questions, determine urgency and client type
- **Source Tracking**: All information is traced back to specific pages and documents
- **Context-Aware Responses**: AI responses incorporate conversation history and document context
- **Automatic Document Anonymization**: Sensitive information in documents is automatically anonymized

## Architecture

The application is built with a modular architecture:

- **Frontend**: Streamlit-based user interface
- **Core RAG Engine**: Document search and context-aware response generation
- **External Services**: Azure OpenAI for embeddings and completion, Pinecone for vector storage
- **Data Layer**: Document and email repositories
- **Anonymization Engine**: Automatically detects and anonymizes sensitive information

For a detailed architectural diagram, see [architecture_diagram.md](architecture_diagram.md).

## Installation

### Prerequisites

- Python 3.9+
- Azure OpenAI API access
- Pinecone API access
- Hugging Face Transformers library
- SSL certificates (if running in corporate environment with SSL inspection)

### Setup

1. Clone the repository
```bash
git clone https://github.com/fedegu94/Hackaton-agentic-RAG.git
cd Hackaton-agentic-RAG
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys
```
# Azure OpenAI Configuration for Chat
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure OpenAI Configuration for Embeddings
AZURE_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.openai.azure.com/
AZURE_EMBEDDING_API_KEY=your-embedding-api-key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_EMBEDDING_API_VERSION=2023-05-15

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=compliance50

# SSL Configuration (for corporate environments)
SSL_CERT_PATH=/path/to/your/certificate.pem
```

You can find a more detailed example in the `.env.example` file.

### SSL Certificate Configuration

If you're running this application in a corporate environment with SSL inspection (such as Zscaler):

1. **Set the SSL certificate path** in your `.env` file:
   ```
   SSL_CERT_PATH=/path/to/your/certificate.pem
   ```

2. **Common certificate locations** that the application will check automatically:
   - User-provided path in `SSL_CERT_PATH` environment variable
   - User's home directory: `~/zscaler-ca-bundle.pem`
   - Project's `certs` folder: `certs/zscaler-ca-bundle.pem`
   - Standard system certificate store via `certifi`

For more details on SSL certificate configuration and troubleshooting, see [ssl_certificates.md](docs/ssl_certificates.md).

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### Chat Interface

1. Select "💬 Chat Normativa" from the sidebar
2. Type your regulatory question in the chat input
3. View AI responses with referenced sources
4. Create new chat sessions using the sidebar controls

### Email Interface

1. Select "📧 Gestione Email" from the sidebar
2. Choose an email to analyze from the dropdown
3. Click "Generate Response" to create an AI-powered draft response
4. View the sources used to generate the response

### Automatic Document Anonymization

The system automatically anonymizes documents placed in the `data/emails/documents` folder:

1. Add your email documents (text files) to the `data/emails/documents` folder
2. The system will automatically process them and create anonymized versions in `data/emails/anonymized`
3. Anonymized versions will mask sensitive information such as:
   - Personal names
   - Email addresses
   - Phone numbers
   - IBAN numbers
   - Fiscal codes
   - Organization names
   - Locations

You can also manually trigger the anonymization process:

```bash
python src/anonymize_service.py
```

To continuously watch for new documents and anonymize them:

```bash
python src/anonymize_service.py --watch
```

You can also specify a custom check interval (in seconds):

```bash
python src/anonymize_service.py --watch --interval 30
```

## Files Structure

```
├── app.py                   # Main application entry point
├── config.py                # Configuration & client setup
├── page_chat.py             # Chat interface implementation
├── page_emails.py           # Email management interface
├── rag_system.py            # Core RAG engine
├── pdf_query_system.py      # PDF document querying system
├── anonymize_mails.py       # Document anonymization engine
├── anonymize_service.py     # CLI tool for anonymization
├── requirements.txt         # Python dependencies
├── architectural_scheme_moca.png # Architecture diagram image
├── architecture_diagram.md  # System architecture documentation
├── CHANGELOG.md             # Version history and changes
├── README.md                # This documentation
└── data/                    # Data directory
    ├── emails/              # Email data
    │   ├── anonymized/      # Anonymized email versions
    │   └── documents/       # Original email documents
    ├── embeddings/          # Vector embeddings 
    │   ├── chunks_with_embeddings_metadata.json
    │   └── chunks_with_embeddings.pkl
    └── pdf_documents/       # PDF knowledge base
└── docs/                    # Documentation
    ├── api_reference.md
    ├── architecture.md
    ├── ssl_certificates.md
    └── user_guide.md
└── tests/                   # Test files
    ├── test_anonymization.py
    └── test_system.py
```

## Development

### Adding New Documents

To expand the knowledge base with new regulatory documents:
1. Add PDF files to the `data/pdf_documents` folder
2. Process them for embedding using the PDF processing module
3. Update the Pinecone vector database with new embeddings

### Customizing Responses

To modify AI response behavior:
1. Edit the system prompts in `rag_system.py`
2. Adjust the number of retrieved documents through the UI sliders
3. Modify response templates in the core RAG engine

### Customizing Anonymization

To modify the anonymization behavior:
1. Edit the patterns and anonymization rules in `anonymize_mails.py`
2. Adjust the entity types and replacements in the NER pipeline configuration
3. Add custom regex patterns for domain-specific anonymization needs

### Customizing SSL Certificate Handling

To modify the SSL certificate handling for corporate environments:
1. Edit the `find_ssl_cert()` function in `config.py` to add additional certificate paths
2. Modify the certificate validation settings to match your organization's requirements
3. For development in environments with SSL inspection, you may need to:
   - Export your organization's root certificates (e.g., Zscaler certificates)
   - Place them in one of the looked-up paths or specify via environment variable
   - Test API connectivity with tools like `curl` using your certificate

### Testing

The application includes test files for various components:
1. Run the anonymization tests to verify sensitive data detection:
```bash
python -m unittest tests/test_anonymization.py
```

2. Run the system tests to verify RAG functionality:
```bash
python -m unittest tests/test_system.py
```

## Contributors

@fedegu95
@pitzus-ey
@LorenzoMartemucci
@AlbertoMancosu
@MichelePolimeni

---

*MOCA AI Normative Consultant is a project developed as part of the AI Academy Hackathon.*

*Last updated: June 30, 2025*
