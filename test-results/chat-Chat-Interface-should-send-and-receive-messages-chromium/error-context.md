# Page snapshot

```yaml
- banner:
  - button "Deploy"
  - button
- alert:
  - paragraph:
    - strong: streamlit.errors.StreamlitSetPageConfigMustBeFirstCommandError
    - text: ":"
    - code: set_page_config()
    - text: can only be called once per app page, and must be called as the first Streamlit command in your script.
  - paragraph:
    - text: For more information refer to the
    - link "docs":
      - /url: https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
    - text: .
  - text: "Traceback:"
  - code: File "/Users/Sour/basic-chat/app.py", line 969, in <module> main() File "/Users/Sour/basic-chat/app.py", line 890, in main st.set_page_config( File "/Users/Sour/.pyenv/versions/3.11.9/lib/python3.11/site-packages/streamlit/runtime/metrics_util.py", line 409, in wrapped_func result = non_optional_func(*args, **kwargs) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/Users/Sour/.pyenv/versions/3.11.9/lib/python3.11/site-packages/streamlit/commands/page_config.py", line 273, in set_page_config ctx.enqueue(msg) File "/Users/Sour/.pyenv/versions/3.11.9/lib/python3.11/site-packages/streamlit/runtime/scriptrunner_utils/script_run_context.py", line 180, in enqueue raise StreamlitSetPageConfigMustBeFirstCommandError()
```