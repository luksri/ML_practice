TMA AI agents:
    Purpose:
            - helpful in reading and analyzing clinical protocols so that we one can look into specific information. For example: exclusion and inclusion criteria.
    
    usage: 
        - Upload a PDF file (only) to begin with. This may take time as the background process of creating a vector store
        - ask anything through prompt:
            example: "extract each and every exclusion and inclusion criteria in a human readable format. convert the text to sql queries"
    

To run the application:
    - App is devloped using PHI and GROQ platforms.
    - follow https://docs.agno.com/vectordb/pgvector to create postgres db
    - create api keys and store it in '.env' file
    - run the command: streamlit run app.py