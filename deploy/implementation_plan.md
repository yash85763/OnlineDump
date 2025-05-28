The error occurs because the PDF database ID is not being stored correctly in the session state during processing. Here are the fixes and comprehensive checks needed:

## **1. Fix PDF Database ID Storage Issue:**

**In `process_pdf_enhanced()` function, after the enhanced handler call, add debugging and fix the storage:**

```python
if result.get('success'):
    # FIX: Store processing information with proper checks
    pdf_id = result.get('pdf_id')
    if pdf_id:
        st.session_state.pdf_database_ids[pdf_name] = pdf_id
        st.session_state.processing_messages[pdf_name].append(f"✅ PDF stored in database with ID: {pdf_id}")
    else:
        st.session_state.processing_messages[pdf_name].append("⚠️ Warning: PDF processed but no database ID returned")
        # Check if it's a duplicate
        if result.get('duplicate'):
            existing_id = result.get('existing_record', {}).get('id')
            if existing_id:
                st.session_state.pdf_database_ids[pdf_name] = existing_id
                st.session_state.processing_messages[pdf_name].append(f"📋 Using existing database ID: {existing_id}")
```

## **2. Add Comprehensive Service Checks:**

**Add this function after `initialize_session_state()`:**

```python
def check_all_services():
    """Check all required services and return status"""
    services_status = {
        'database': {'status': False, 'message': 'Not checked'},
        'obfuscation': {'status': False, 'message': 'Not checked'},
        'pdf_handler': {'status': False, 'message': 'Not checked'},
        'contract_analyzer': {'status': False, 'message': 'Not checked'}
    }
    
    # Check Database
    try:
        from config.database import check_database_connection
        if check_database_connection():
            services_status['database'] = {'status': True, 'message': 'Connected'}
        else:
            services_status['database'] = {'status': False, 'message': 'Connection failed'}
    except ImportError:
        services_status['database'] = {'status': False, 'message': 'Module not found'}
    except Exception as e:
        services_status['database'] = {'status': False, 'message': f'Error: {str(e)}'}
    
    # Check Obfuscation Service
    try:
        from services.obfuscation import ContentObfuscator
        obfuscator = ContentObfuscator()
        services_status['obfuscation'] = {'status': True, 'message': 'Available'}
    except ImportError:
        services_status['obfuscation'] = {'status': False, 'message': 'Module not found'}
    except Exception as e:
        services_status['obfuscation'] = {'status': False, 'message': f'Error: {str(e)}'}
    
    # Check PDF Handler
    try:
        from utils.enhanced_pdf_handler import EnhancedPDFHandler
        handler = EnhancedPDFHandler(enable_obfuscation=False, enable_database=False)
        services_status['pdf_handler'] = {'status': True, 'message': 'Available'}
    except ImportError:
        services_status['pdf_handler'] = {'status': False, 'message': 'Module not found'}
    except Exception as e:
        services_status['pdf_handler'] = {'status': False, 'message': f'Error: {str(e)}'}
    
    # Check Contract Analyzer
    try:
        from contract_analyzer import ContractAnalyzer
        analyzer = ContractAnalyzer()
        services_status['contract_analyzer'] = {'status': True, 'message': 'Available'}
    except ImportError:
        services_status['contract_analyzer'] = {'status': False, 'message': 'Module not found'}
    except Exception as e:
        services_status['contract_analyzer'] = {'status': False, 'message': f'Error: {str(e)}'}
    
    return services_status
```

## **3. Update Session State Initialization:**

**Replace the database initialization part in `initialize_session_state()` with:**

```python
def initialize_session_state():
    """Initialize all session state variables"""
    # Service checks
    if 'services_checked' not in st.session_state:
        st.session_state.services_status = check_all_services()
        st.session_state.services_checked = True
    
    # Database initialization
    if 'database_initialized' not in st.session_state:
        if st.session_state.services_status['database']['status']:
            try:
                initialize_database()
                st.session_state.database_initialized = True
                st.session_state.database_status = "Connected and initialized"
            except Exception as e:
                st.session_state.database_initialized = False
                st.session_state.database_status = f"Initialization failed: {str(e)}"
        else:
            st.session_state.database_initialized = False
            st.session_state.database_status = st.session_state.services_status['database']['message']
    
    # ... rest of session_vars remains the same
```

## **4. Enhanced Feedback Form with Better Error Handling:**

**Replace the feedback submission section in `render_feedback_form()` with:**

```python
if submitted:
    # Validate that at least some feedback is provided
    if (form_number_correct == "Select..." and pi_clause_correct == "Select..." and 
        ci_clause_correct == "Select..." and summary_quality == "Select..." and 
        not general_feedback.strip()):
        st.error("Please provide at least some feedback before submitting.")
        return
    
    # Enhanced PDF ID retrieval with debugging
    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
    
    if not pdf_id:
        # Debug information
        st.error("❌ Cannot submit feedback - PDF not found in database")
        
        with st.expander("🔧 Debug Information", expanded=False):
            st.write("**Available PDF Database IDs:**")
            st.write(st.session_state.pdf_database_ids)
            st.write(f"**Looking for PDF:** {pdf_name}")
            st.write(f"**Current PDF:** {st.session_state.current_pdf}")
            st.write(f"**File stem:** {file_stem}")
            
            # Check if database is working
            if not st.session_state.database_initialized:
                st.write("**Database Status:** Not initialized")
            else:
                st.write("**Database Status:** Connected")
        
        st.info("💡 Try reprocessing the PDF to generate a database entry.")
        return
    
    # Check database connection before submitting
    if not st.session_state.database_initialized:
        st.error("❌ Database not available. Cannot store feedback.")
        return
    
    # Prepare feedback data (existing code remains the same)
    # ... feedback_data preparation ...
    
    try:
        feedback_id = store_feedback_data(feedback_data)
        st.success("🎉 Thank you for your valuable feedback! It helps us improve our analysis.")
        st.session_state.feedback_submitted[feedback_key] = True
        st.balloons()
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to save feedback: {str(e)}")
        
        # Additional error context
        with st.expander("🔧 Error Details", expanded=False):
            st.write(f"**Error Type:** {type(e).__name__}")
            st.write(f"**Error Message:** {str(e)}")
            st.write(f"**PDF ID:** {pdf_id}")
            st.write(f"**Database Status:** {st.session_state.database_status}")
```

## **5. Enhanced Sidebar with Service Status:**

**Update the sidebar section in `main()` with:**

```python
with st.sidebar:
    st.header("🔧 System Status")
    
    # Service status checks
    services = st.session_state.get('services_status', {})
    
    for service_name, service_info in services.items():
        status = service_info['status']
        message = service_info['message']
        
        if status:
            st.markdown(f"✅ **{service_name.replace('_', ' ').title()}:** {message}")
        else:
            st.markdown(f"❌ **{service_name.replace('_', ' ').title()}:** {message}")
    
    # Overall database status
    if st.session_state.database_initialized:
        st.success("✅ Database Ready")
    else:
        st.error(f"❌ Database: {st.session_state.database_status}")
    
    # Refresh services button
    if st.button("🔄 Refresh Services"):
        st.session_state.services_status = check_all_services()
        st.rerun()
    
    # ... rest of sidebar content ...
```

## **6. Add Process Validation:**

**Before processing any PDF, add this check in the grid selection section:**

```python
# Before processing, check if all services are available
services_ok = all(service['status'] for service in st.session_state.services_status.values())

if not services_ok:
    st.error("❌ Cannot process PDF - some services are unavailable")
    failed_services = [name for name, info in st.session_state.services_status.items() if not info['status']]
    st.write(f"Failed services: {', '.join(failed_services)}")
    return

# Then proceed with processing...
```

These changes will:
- ✅ **Fix the database ID storage issue**
- ✅ **Add comprehensive service checks**
- ✅ **Provide detailed error debugging**
- ✅ **Show service status in sidebar**
- ✅ **Prevent processing when services are down**
- ✅ **Give users clear feedback about what's wrong**

The feedback submission should now work correctly, and you'll have full visibility into any service issues.​​​​​​​​​​​​​​​​