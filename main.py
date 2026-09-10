import os
import sys


import warnings
warnings.filterwarnings("ignore")


from orchestration import build_graph

def main():
    print("Loading AI Workers and connecting to Vector DB...")
    

    app = build_graph()
    
    print("Ready!\n")
    print("Type 'exit' or 'quit' to close the application.")
    
    while True:
        try:
            print("\n--------------------------------------------------")
            user_question = input("\nAsk a legal question:> ").strip()
            
            if user_question.lower() in ['exit', 'quit', 'q']:
                print("\nShutting down pipeline. Goodbye!")
                break
                
            if not user_question:
                continue
                
            print("\n")
            initial_state = {"question": user_question}
            

            final_state = app.invoke(initial_state)
            
            print("\n=== FINAL ANSWER ===")
            print(final_state.get("final_answer", "No answer was generated."))
            
        except KeyboardInterrupt:
            print("\n\nShutting down pipeline. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
