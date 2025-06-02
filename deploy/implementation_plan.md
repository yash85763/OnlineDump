To design a database schema for a contract analysis application that allows users to upload PDF contracts, process and analyze them, and provide feedback, while also handling user authentication and session management, we need to identify the key entities and their relationships. Below, I’ll outline the required tables, their columns, and how they are linked, followed by a textual description of the database diagram since I cannot draw it here.
Required Tables and Columns
1. users Table

This table stores user information for authentication and session management.

    Columns:
        id (Primary Key, e.g., INTEGER): Unique identifier for each user.
        username (e.g., VARCHAR): User’s login name.
        password_hash (e.g., VARCHAR): Hashed password for security.
        session_id (e.g., VARCHAR, unique): Tracks the user’s current session.
        created_at (e.g., DATETIME): Timestamp of account creation.
        last_active (e.g., DATETIME): Timestamp of last activity.

2. pdfs Table

This table stores details about uploaded PDF contracts.

    Columns:
        id (Primary Key, e.g., INTEGER): Unique identifier for each PDF.
        pdf_name (e.g., VARCHAR): Name of the PDF file.
        file_hash (e.g., VARCHAR, unique): Hash for deduplication.
        upload_date (e.g., DATETIME): When the PDF was uploaded.
        processed_date (e.g., DATETIME): When the PDF was processed.
        layout (e.g., VARCHAR): Layout type (e.g., single-column).
        original_word_count (e.g., INTEGER): Word count before processing.
        original_page_count (e.g., INTEGER): Page count before processing.
        parsability (e.g., FLOAT): Measure of how parseable the PDF is.
        final_word_count (e.g., INTEGER): Word count after processing.
        final_page_count (e.g., INTEGER): Page count after processing.
        avg_words_per_page (e.g., FLOAT): Average words per page.
        raw_content (e.g., JSON): Raw extracted content (pages and paragraphs).
        final_content (e.g., TEXT): Processed content.
        obfuscation_applied (e.g., BOOLEAN): Whether obfuscation was applied.
        pages_removed_count (e.g., INTEGER): Number of pages removed.
        paragraphs_obfuscated_count (e.g., INTEGER): Number of paragraphs obfuscated.
        obfuscation_summary (e.g., JSON): Summary of obfuscation details.
        uploaded_by (Foreign Key to users.id, e.g., INTEGER): Links to the user who uploaded the PDF.

3. analyses Table

This table stores the results of contract analyses.

    Columns:
        id (Primary Key, e.g., INTEGER): Unique identifier for each analysis.
        pdf_id (Foreign Key to pdfs.id, e.g., INTEGER): Links to the analyzed PDF.
        analysis_date (e.g., DATETIME): When the analysis was performed.
        version (e.g., VARCHAR): Version of the analysis (e.g., "v1.0").
        form_number (e.g., VARCHAR): Contract form number, if applicable.
        pi_clause (e.g., TEXT): Personal information clause extracted.
        ci_clause (e.g., TEXT): Confidential information clause extracted.
        data_usage_mentioned (e.g., BOOLEAN): Whether data usage is mentioned.
        data_limitations_exists (e.g., BOOLEAN): Whether data limitations exist.
        summary (e.g., TEXT): Summary of the analysis.
        raw_json (e.g., JSON): Raw analysis output.
        processed_by (e.g., VARCHAR): Who or what processed it (e.g., "AI").
        processing_time (e.g., FLOAT): Time taken for analysis (in seconds).

4. clauses Table

This table stores specific clauses extracted from the contracts, including their positions for highlighting.

    Columns:
        id (Primary Key, e.g., INTEGER): Unique identifier for each clause.
        analysis_id (Foreign Key to analyses.id, e.g., INTEGER): Links to the analysis.
        clause_type (e.g., VARCHAR): Type of clause (e.g., "PI", "CI").
        clause_text (e.g., TEXT): Text of the clause.
        page_number (e.g., INTEGER): Page where the clause appears.
        paragraph_index (e.g., INTEGER): Paragraph index on that page (0-based).
        clause_order (e.g., INTEGER): Order of the clause in the contract.

5. feedback Table

This table stores user feedback on the PDF’s analysis.

    Columns:
        id (Primary Key, e.g., INTEGER): Unique identifier for each feedback entry.
        pdf_id (Foreign Key to pdfs.id, e.g., INTEGER): Links to the PDF being reviewed.
        feedback_date (e.g., DATETIME): When the feedback was provided.
        general_feedback (e.g., TEXT): User’s feedback comments.
        user_id (Foreign Key to users.id, e.g., INTEGER): Links to the user who provided feedback.

Table Relationships

The tables are linked via foreign keys to establish the following relationships:

    Users to PDFs: One-to-Many
        A user can upload multiple PDFs, but each PDF is uploaded by one user.
        pdfs.uploaded_by references users.id.
    PDFs to Analyses: One-to-Many
        A PDF can have multiple analyses (e.g., different versions), but each analysis pertains to one PDF.
        analyses.pdf_id references pdfs.id.
    Analyses to Clauses: One-to-Many
        An analysis can identify multiple clauses, but each clause belongs to one analysis.
        clauses.analysis_id references analyses.id.
    PDFs to Feedback: One-to-Many
        A PDF can have multiple feedback entries, but each feedback entry pertains to one PDF.
        feedback.pdf_id references pdfs.id.
    Users to Feedback: One-to-Many
        A user can provide multiple feedback entries, but each feedback entry is from one user.
        feedback.user_id references users.id.

Database Diagram Description

Since I cannot provide a visual diagram, here’s a textual representation:

    users Table:
        Central entity with id as the primary key.
        Connected to pdfs via uploaded_by (FK) and to feedback via user_id (FK).
    pdfs Table:
        Has id as the primary key and uploaded_by as a foreign key to users.id.
        Connected to analyses via pdf_id (FK) and to feedback via pdf_id (FK).
    analyses Table:
        Has id as the primary key and pdf_id as a foreign key to pdfs.id.
        Connected to clauses via analysis_id (FK).
    clauses Table:
        Has id as the primary key and analysis_id as a foreign key to analyses.id.
        No outgoing connections.
    feedback Table:
        Has id as the primary key, pdf_id as a foreign key to pdfs.id, and user_id as a foreign key to users.id.

The flow is:

    Users → PDFs (users upload PDFs).
    PDFs → Analyses (PDFs are analyzed).
    Analyses → Clauses (analyses extract clauses).
    Users and PDFs → Feedback (users provide feedback on PDFs).

Additional Notes

    The clauses table includes page_number and paragraph_index to support clause highlighting by linking back to the raw_content JSON in the pdfs table.
    Feedback is linked to pdfs rather than analyses to reflect general feedback on a PDF’s analysis, though linking to analyses could be an alternative for version-specific feedback.
    Session management is simplified with session_id in the users table; a separate sessions table could be added for more complex needs.

This schema supports user authentication, PDF upload and processing, contract analysis, clause extraction with highlighting, and feedback collection, meeting all specified requirements.
