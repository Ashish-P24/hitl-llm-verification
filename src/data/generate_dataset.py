import os
import pandas as pd


# ---------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------

DATASET = []


def add_examples(category, examples):
    """
    Add examples to the dataset.

    Each example:
    (question, response, risk_score, needs_human_review)
    """

    for question, response, risk_score, needs_review in examples:
        DATASET.append({
            "question": question,
            "response": response,
            "category": category,
            "risk_score": risk_score,
            "needs_human_review": needs_review,
        })


# ---------------------------------------------------------
# General Knowledge
# ---------------------------------------------------------

add_examples("General Knowledge", [
    ("What is the capital of France?",
     "The capital of France is Paris.", 0.05, 0),

    ("How many continents are there?",
     "There are seven commonly recognized continents.", 0.05, 0),

    ("What is the largest ocean?",
     "The Pacific Ocean is the largest ocean on Earth.", 0.05, 0),

    ("How many days are in a leap year?",
     "A leap year has 366 days.", 0.04, 0),

    ("What is the currency of Japan?",
     "The currency of Japan is the Japanese yen.", 0.05, 0),

    ("What is the boiling point of water at sea level?",
     "Water boils at approximately 100 degrees Celsius at standard atmospheric pressure.", 0.06, 0),

    ("Which planet is known as the Red Planet?",
     "Mars is commonly known as the Red Planet.", 0.04, 0),

    ("What is the largest mammal?",
     "The blue whale is the largest known mammal.", 0.05, 0),

    ("How many sides does a hexagon have?",
     "A hexagon has six sides.", 0.03, 0),

    ("What language is primarily spoken in Brazil?",
     "Portuguese is the primary language spoken in Brazil.", 0.06, 0),

    ("What is the tallest mountain above sea level?",
     "Mount Everest is the tallest mountain above sea level.", 0.07, 0),

    ("What is the smallest prime number?",
     "The smallest prime number is 2.", 0.03, 0),

    ("Which gas do humans need to breathe?",
     "Humans need oxygen for normal respiration.", 0.04, 0),

    ("How many hours are in one day?",
     "There are 24 hours in one day.", 0.03, 0),

    ("What is the capital of Australia?",
     "The capital of Australia is Canberra.", 0.05, 0),

    ("Which animal is commonly called the king of the jungle?",
     "The lion is commonly given this description.", 0.08, 0),

    ("How many planets are in the Solar System?",
     "There are eight recognized planets in the Solar System.", 0.05, 0),

    ("What is Earth's natural satellite?",
     "The Moon is Earth's natural satellite.", 0.04, 0),

    ("What color results from mixing blue and yellow paint?",
     "Mixing blue and yellow paint commonly produces green.", 0.06, 0),

    ("What is the freezing point of water?",
     "Water freezes at approximately 0 degrees Celsius at standard atmospheric pressure.", 0.05, 0),
])


# ---------------------------------------------------------
# Mathematics
# ---------------------------------------------------------

add_examples("Mathematics", [
    ("What is 15 multiplied by 8?",
     "15 multiplied by 8 is 120.", 0.03, 0),

    ("What is 25 percent of 200?",
     "25 percent of 200 is 50.", 0.03, 0),

    ("What is the derivative of x squared?",
     "The derivative of x squared is 2x.", 0.04, 0),

    ("What is 144 divided by 12?",
     "144 divided by 12 is 12.", 0.03, 0),

    ("What is the square root of 81?",
     "The square root of 81 is 9.", 0.03, 0),

    ("What is 7 multiplied by 9?",
     "7 multiplied by 9 is 63.", 0.03, 0),

    ("What is the integral of 2x?",
     "The integral of 2x is x squared plus a constant.", 0.06, 0),

    ("What is the area of a square with side 5?",
     "The area is 25 square units.", 0.04, 0),

    ("What is 10 raised to the power of 3?",
     "10 raised to the power of 3 is 1000.", 0.03, 0),

    ("Solve 2x = 10.",
     "The solution is x = 5.", 0.03, 0),

    ("What is the average of 10, 20, and 30?",
     "The average is 20.", 0.03, 0),

    ("What is 5 factorial?",
     "5 factorial is 120.", 0.03, 0),

    ("What is the probability of getting heads on a fair coin?",
     "The probability is one half, or 50 percent.", 0.05, 0),

    ("What is the slope of y = 3x + 2?",
     "The slope is 3.", 0.04, 0),

    ("What is 20 minus 7?",
     "20 minus 7 is 13.", 0.02, 0),

    ("What is 9 squared?",
     "9 squared is 81.", 0.02, 0),

    ("What is the perimeter of a square with side 4?",
     "The perimeter is 16 units.", 0.03, 0),

    ("What is 2 to the fifth power?",
     "2 to the fifth power is 32.", 0.03, 0),

    ("What is the value of pi approximately?",
     "Pi is approximately 3.14159.", 0.05, 0),

    ("What is the sum of the angles in a triangle?",
     "The interior angles of a triangle sum to 180 degrees.", 0.04, 0),
])


# ---------------------------------------------------------
# Science
# ---------------------------------------------------------

add_examples("Science", [
    ("What is the chemical formula for water?",
     "The chemical formula for water is H2O.", 0.04, 0),

    ("What is photosynthesis?",
     "Photosynthesis is the process through which plants convert light energy into chemical energy.", 0.08, 0),

    ("What is the largest planet in our solar system?",
     "Jupiter is the largest planet in our solar system.", 0.04, 0),

    ("What gas do plants absorb during photosynthesis?",
     "Plants absorb carbon dioxide during photosynthesis.", 0.05, 0),

    ("What is the speed of light in vacuum approximately?",
     "The speed of light in vacuum is approximately 300,000 kilometers per second.", 0.08, 0),

    ("What is DNA?",
     "DNA is a molecule that carries genetic information in living organisms.", 0.08, 0),

    ("What force keeps planets in orbit?",
     "Gravity is the primary force responsible for planetary orbits.", 0.07, 0),

    ("What is the atomic number of hydrogen?",
     "Hydrogen has atomic number 1.", 0.04, 0),

    ("What is an atom?",
     "An atom is the basic unit of an element that retains the chemical properties of that element.", 0.07, 0),

    ("What is evaporation?",
     "Evaporation is the process in which a liquid changes into a gas at its surface.", 0.06, 0),

    ("What is the human body temperature normally around?",
     "Normal human body temperature is commonly around 37 degrees Celsius, although it varies.", 0.10, 0),

    ("What is an ecosystem?",
     "An ecosystem consists of living organisms and their physical environment interacting with one another.", 0.07, 0),

    ("Can a scientific theory ever change?",
     "Scientific theories can be refined or modified when strong new evidence becomes available.", 0.12, 0),

    ("Is a new scientific discovery definitely correct?",
     "A new discovery requires independent evidence and validation before its conclusions can be considered reliable.", 0.45, 1),

    ("What will the result of this unpublished experiment be?",
     "The exact result cannot be known with certainty before the experiment is conducted.", 0.70, 1),

    ("Can this experimental treatment definitely cure the disease?",
     "A treatment cannot be guaranteed to cure a disease without appropriate clinical evidence.", 0.85, 1),

    ("Is this new scientific claim definitely true?",
     "The claim should be evaluated against evidence and independent research before being accepted.", 0.65, 1),

    ("Will this chemical reaction definitely occur under these conditions?",
     "The outcome depends on the specific substances and experimental conditions.", 0.60, 1),

    ("Is this laboratory result proof of a new theory?",
     "A single laboratory result is generally insufficient to establish a scientific theory.", 0.55, 1),

    ("Can scientists predict exactly what will happen in a complex ecosystem?",
     "Complex ecosystems contain many interacting variables, making exact predictions difficult.", 0.60, 1),
])


# ---------------------------------------------------------
# Technology
# ---------------------------------------------------------

add_examples("Technology", [
    ("What does CPU stand for?",
     "CPU stands for Central Processing Unit.", 0.05, 0),

    ("What is Python used for?",
     "Python is commonly used for software development, data analysis, machine learning, automation, and web development.", 0.08, 0),

    ("What is RAM?",
     "RAM is temporary computer memory used to store data that active programs need.", 0.05, 0),

    ("What is an operating system?",
     "An operating system manages computer hardware and provides services for applications.", 0.06, 0),

    ("What is HTTP?",
     "HTTP is a protocol used for communication between clients and web servers.", 0.06, 0),

    ("What is a database?",
     "A database is an organized collection of data that can be stored and accessed electronically.", 0.06, 0),

    ("What is Git used for?",
     "Git is a distributed version control system used to track changes in files and collaborate on software projects.", 0.06, 0),

    ("What is an API?",
     "An API is an interface that allows software components to communicate with each other.", 0.06, 0),

    ("What is machine learning?",
     "Machine learning is a field in which algorithms learn patterns from data to make predictions or decisions.", 0.10, 0),

    ("What is cloud computing?",
     "Cloud computing provides computing resources such as storage and processing over networks.", 0.08, 0),

    ("What is encryption?",
     "Encryption transforms information into a form that is difficult to read without the appropriate key.", 0.08, 0),

    ("What is a compiler?",
     "A compiler translates source code into another form, often machine code or intermediate code.", 0.07, 0),

    ("Is this software completely secure?",
     "No software can generally be guaranteed to be completely secure because vulnerabilities may remain undiscovered.", 0.65, 1),

    ("Can this application never be hacked?",
     "Absolute protection against every possible attack cannot be guaranteed.", 0.70, 1),

    ("Will this algorithm always outperform other algorithms?",
     "Performance depends on the dataset, implementation, and evaluation conditions.", 0.60, 1),

    ("Is this AI model completely unbiased?",
     "No model can generally be guaranteed to be completely free of bias.", 0.65, 1),

    ("Can this system handle every possible user input?",
     "A system's ability to handle inputs depends on its design, validation, and operating conditions.", 0.55, 1),

    ("Will this code definitely work in production?",
     "Production behavior depends on the environment, dependencies, configuration, and testing.", 0.55, 1),

    ("Can this server never go down?",
     "No production system can generally guarantee zero downtime.", 0.65, 1),

    ("Is this cybersecurity configuration guaranteed to prevent attacks?",
     "Security controls reduce risk but cannot guarantee prevention of every possible attack.", 0.70, 1),
])


# ---------------------------------------------------------
# Medical
# ---------------------------------------------------------

add_examples("Medical", [
    ("What is the function of the heart?",
     "The heart pumps blood throughout the body.", 0.10, 0),

    ("What is a vitamin?",
     "A vitamin is an organic compound needed by the body in relatively small amounts for normal functioning.", 0.10, 0),

    ("What is the purpose of vaccination?",
     "Vaccination trains the immune system to recognize specific pathogens or their components.", 0.12, 0),

    ("What is blood pressure?",
     "Blood pressure is the pressure exerted by circulating blood against blood vessel walls.", 0.10, 0),

    ("What does the immune system do?",
     "The immune system helps the body identify and respond to harmful pathogens and abnormal cells.", 0.10, 0),

    ("What is dehydration?",
     "Dehydration occurs when the body loses more fluid than it takes in.", 0.10, 0),

    ("Why is sleep important?",
     "Sleep supports physical recovery, cognitive function, and several regulatory processes.", 0.12, 0),

    ("What is a balanced diet?",
     "A balanced diet provides a variety of nutrients in appropriate amounts.", 0.10, 0),

    ("What is the role of red blood cells?",
     "Red blood cells primarily transport oxygen through the bloodstream.", 0.08, 0),

    ("What is cholesterol?",
     "Cholesterol is a lipid that plays important roles in the body but can be harmful at abnormal levels.", 0.15, 0),

    ("What is the purpose of antibiotics?",
     "Antibiotics are medicines used to treat certain bacterial infections.", 0.12, 0),

    ("What is a fever?",
     "A fever is an elevation of body temperature that can occur as part of the body's response to illness.", 0.12, 0),

    ("What is the best treatment for chest pain?",
     "The appropriate response to chest pain depends on its cause and severity, and serious causes may require urgent medical assessment.", 0.95, 1),

    ("Should I stop taking my prescribed medication?",
     "Medication should not generally be stopped without discussing the decision with an appropriate healthcare professional.", 0.98, 1),

    ("What is the exact cause of my symptoms?",
     "An exact diagnosis cannot reliably be determined from limited symptoms alone.", 0.92, 1),

    ("Can I double my medication dose?",
     "Medication dosage should not be changed without appropriate professional guidance.", 0.96, 1),

    ("Is this symptom definitely cancer?",
     "A symptom alone cannot establish a cancer diagnosis.", 0.94, 1),

    ("Can this supplement cure my disease?",
     "A supplement should not be assumed to cure a disease without appropriate clinical evidence.", 0.90, 1),

    ("Should I ignore severe abdominal pain?",
     "Severe or worsening abdominal pain may require medical evaluation rather than being ignored.", 0.92, 1),

    ("Can I diagnose myself from this online information?",
     "Online information cannot reliably replace an individualized medical evaluation.", 0.88, 1),
])


# ---------------------------------------------------------
# Legal
# ---------------------------------------------------------

add_examples("Legal", [
    ("What is a contract?",
     "A contract is a legally recognized agreement between parties that creates obligations.", 0.12, 0),

    ("What is a trademark?",
     "A trademark is a sign or identifier used to distinguish goods or services.", 0.10, 0),

    ("What is copyright?",
     "Copyright provides legal protection for certain original creative works.", 0.10, 0),

    ("What is a patent?",
     "A patent provides certain exclusive rights over an invention for a defined period under applicable law.", 0.12, 0),

    ("What is a witness?",
     "A witness is a person who provides information or evidence about matters relevant to a legal proceeding.", 0.12, 0),

    ("What is a legal agreement?",
     "A legal agreement is an arrangement between parties that may create legally enforceable obligations.", 0.15, 0),

    ("What is arbitration?",
     "Arbitration is a dispute resolution process in which a neutral arbitrator considers the parties' positions.", 0.15, 0),

    ("What is a civil case?",
     "A civil case generally concerns disputes between individuals or organizations rather than criminal prosecution.", 0.15, 0),

    ("Can laws vary between countries?",
     "Yes. Laws and legal procedures vary between jurisdictions.", 0.08, 0),

    ("What is a court?",
     "A court is a legal institution that adjudicates disputes and applies relevant law.", 0.08, 0),

    ("What is legal liability?",
     "Legal liability refers to responsibility that may arise under applicable law.", 0.15, 0),

    ("What is a regulation?",
     "A regulation is a rule issued by an authority under an applicable legal framework.", 0.12, 0),

    ("Can I legally use copyrighted software without permission?",
     "Whether use is lawful depends on the applicable license, copyright law, and specific circumstances.", 0.90, 1),

    ("Can I sue someone for this situation?",
     "Whether a lawsuit is appropriate depends on the jurisdiction, facts, evidence, and applicable law.", 0.88, 1),

    ("Is this contract definitely enforceable?",
     "Enforceability depends on the jurisdiction, contract terms, formation, and surrounding circumstances.", 0.85, 1),

    ("Can I ignore a legal notice?",
     "The appropriate response depends on the notice and applicable law, so professional legal advice may be appropriate.", 0.90, 1),

    ("Am I definitely protected by this law?",
     "Legal protection depends on the exact facts, jurisdiction, and applicable legal provisions.", 0.88, 1),

    ("Will I definitely win this case?",
     "No outcome can generally be guaranteed because legal disputes depend on evidence, law, and judicial decisions.", 0.95, 1),

    ("Can I use this contract template without modification?",
     "A template may not address the specific legal requirements and circumstances of every situation.", 0.80, 1),

    ("Is this action definitely legal in my country?",
     "Legality depends on the specific conduct and the laws applicable in the relevant jurisdiction.", 0.88, 1),
])


# ---------------------------------------------------------
# Financial
# ---------------------------------------------------------

add_examples("Financial", [
    ("What is a savings account?",
     "A savings account is a bank account designed primarily for holding money and may pay interest.", 0.10, 0),

    ("What is compound interest?",
     "Compound interest is interest calculated on the initial amount and accumulated interest.", 0.08, 0),

    ("What is inflation?",
     "Inflation refers to a general increase in prices over time, reducing purchasing power.", 0.10, 0),

    ("What is a stock?",
     "A stock represents an ownership interest in a company.", 0.08, 0),

    ("What is a bond?",
     "A bond is generally a debt instrument through which an issuer borrows money from investors.", 0.10, 0),

    ("What is diversification?",
     "Diversification involves spreading investments across different assets or exposures to reduce concentration risk.", 0.12, 0),

    ("What is a mutual fund?",
     "A mutual fund pools money from investors to invest in a portfolio of assets according to its objectives.", 0.10, 0),

    ("What is a budget?",
     "A budget is a plan for managing expected income and expenses.", 0.06, 0),

    ("What is interest?",
     "Interest is the cost of borrowing money or the return earned for lending or depositing money.", 0.08, 0),

    ("What is a credit score?",
     "A credit score is a numerical measure used by financial institutions to assess aspects of credit risk.", 0.12, 0),

    ("What is an index fund?",
     "An index fund generally seeks to track the performance of a specified market index.", 0.10, 0),

    ("What is market capitalization?",
     "Market capitalization is generally calculated by multiplying a company's share price by its outstanding shares.", 0.08, 0),

    ("Will this investment definitely make money?",
     "No investment can generally be guaranteed to make money because investments involve risk.", 0.96, 1),

    ("Should I invest all my savings in this stock?",
     "Investment decisions depend on financial circumstances, risk tolerance, objectives, and diversification.", 0.94, 1),

    ("Will this stock definitely rise tomorrow?",
     "Short-term stock movements cannot be predicted with certainty.", 0.92, 1),

    ("Should I take a large loan for this investment?",
     "Borrowing for investment involves significant financial risk and depends on individual circumstances.", 0.90, 1),

    ("Can cryptocurrency never lose value?",
     "Cryptocurrency values can fluctuate substantially and cannot generally be guaranteed to remain stable.", 0.92, 1),

    ("What is the exact future price of Bitcoin?",
     "The exact future price cannot be reliably known in advance.", 0.90, 1),

    ("Is this financial product guaranteed to be safe?",
     "Financial products involve different risks, and safety cannot generally be guaranteed without examining the specific product.", 0.88, 1),

    ("Should I sell all my investments today?",
     "Such a decision depends on personal financial circumstances, objectives, taxes, and market conditions.", 0.92, 1),
])


# ---------------------------------------------------------
# Current Affairs
# ---------------------------------------------------------

add_examples("Current Affairs", [
    ("What is the role of the United Nations?",
     "The United Nations is an international organization focused on cooperation among member states on global issues.", 0.12, 0),

    ("What is a central bank?",
     "A central bank is an institution responsible for functions such as monetary policy and currency-related responsibilities.", 0.12, 0),

    ("What is a parliamentary election?",
     "A parliamentary election is an election used to choose representatives for a legislature.", 0.10, 0),

    ("What is inflation targeting?",
     "Inflation targeting is a monetary policy framework in which a central bank aims to maintain inflation around a specified target.", 0.15, 0),

    ("What is a trade agreement?",
     "A trade agreement establishes terms governing trade between participating countries or regions.", 0.12, 0),

    ("What is a government budget?",
     "A government budget outlines expected public revenues and expenditures for a specified period.", 0.10, 0),

    ("What is a national election?",
     "A national election is a country-wide electoral process for choosing government representatives.", 0.08, 0),

    ("What is a diplomatic agreement?",
     "A diplomatic agreement is an arrangement reached between governments or other recognized parties.", 0.12, 0),

    ("What is a public referendum?",
     "A referendum allows voters to directly decide on a specified political or policy question.", 0.10, 0),

    ("Why do interest rates change?",
     "Interest rates can change in response to monetary policy decisions and economic conditions.", 0.15, 0),

    ("Why do exchange rates change?",
     "Exchange rates change due to factors including supply and demand, economic conditions, interest rates, and expectations.", 0.15, 0),

    ("Why can economic forecasts change?",
     "Economic forecasts can change when new data, assumptions, or economic conditions emerge.", 0.15, 0),

    ("What is the current repo rate?",
     "The current repo rate depends on the latest monetary policy decision and should be verified using an authoritative current source.", 0.80, 1),

    ("Who is currently the richest person in the world?",
     "Rankings of the world's richest people change over time with asset prices and other factors.", 0.75, 1),

    ("What will the weather be tomorrow?",
     "Tomorrow's weather depends on the location and the latest forecast information.", 0.75, 1),

    ("Who will win the next national election?",
     "An election outcome cannot be known with certainty before the election takes place.", 0.90, 1),

    ("What will the stock market do tomorrow?",
     "Short-term market movements cannot be predicted with certainty.", 0.90, 1),

    ("What will the central bank announce next month?",
     "Future policy decisions cannot be known with certainty before they are officially announced.", 0.85, 1),

    ("Which country will have the strongest economy next year?",
     "Future economic performance depends on many uncertain factors and cannot be determined with certainty.", 0.80, 1),

    ("What will happen to inflation next quarter?",
     "Inflation forecasts are uncertain and depend on future economic conditions and incoming data.", 0.75, 1),
])


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

add_examples("Prediction", [
    ("Will it rain next week?",
     "Weather forecasts can provide estimates, but conditions may change.", 0.60, 1),

    ("Will this company succeed?",
     "The outcome depends on many uncertain business and market factors.", 0.65, 1),

    ("Will this student pass the exam?",
     "The result depends on preparation, exam performance, and other factors.", 0.55, 1),

    ("Will this team win the tournament?",
     "Tournament outcomes depend on future performance and cannot be guaranteed.", 0.65, 1),

    ("Will this startup become profitable?",
     "Profitability depends on factors such as revenue, costs, competition, and execution.", 0.60, 1),

    ("Will the price of gold increase?",
     "Gold prices are influenced by economic and market conditions and cannot be predicted with certainty.", 0.65, 1),

    ("Will AI replace all programmers?",
     "The future impact of AI on software development is uncertain and likely to vary by role and technology.", 0.60, 1),

    ("Will this algorithm always produce the best result?",
     "No algorithm can generally be guaranteed to produce the best result for every possible problem.", 0.70, 1),

    ("Will this product become popular?",
     "Popularity depends on market response, competition, product quality, and other uncertain factors.", 0.60, 1),

    ("Will housing prices fall next year?",
     "Future housing prices depend on economic conditions, supply, demand, and policy.", 0.70, 1),

    ("Will this treatment work for every patient?",
     "Treatment outcomes can vary between individuals and cannot generally be guaranteed.", 0.85, 1),

    ("Will this model always classify new data correctly?",
     "No predictive model can generally guarantee correct classification for every future example.", 0.70, 1),

    ("What is the capital of France?",
     "The capital of France is Paris.", 0.05, 0),

    ("What is 10 plus 20?",
     "10 plus 20 equals 30.", 0.03, 0),

    ("What is the boiling point of water?",
     "Water boils at approximately 100 degrees Celsius at standard atmospheric pressure.", 0.05, 0),

    ("What is the largest planet?",
     "Jupiter is the largest planet in the Solar System.", 0.04, 0),

    ("How many days are in a week?",
     "There are seven days in a week.", 0.02, 0),

    ("What is the square root of 64?",
     "The square root of 64 is 8.", 0.03, 0),

    ("What does HTML stand for?",
     "HTML stands for HyperText Markup Language.", 0.04, 0),

    ("What is the derivative of x squared?",
     "The derivative of x squared is 2x.", 0.04, 0),
])


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

def build_dataset():
    df = pd.DataFrame(DATASET)

    # Text length features
    df["question_length"] = df["question"].str.len()
    df["response_length"] = df["response"].str.len()

    # Response word count
    df["word_count"] = (
        df["response"]
        .str.split()
        .str.len()
    )

    # Uncertainty indicators
    uncertainty_words = [
        "maybe",
        "might",
        "possibly",
        "likely",
        "probably",
        "could",
        "uncertain",
        "depends",
        "cannot",
        "generally",
    ]

    df["uncertainty_count"] = df["response"].apply(
        lambda text: sum(
            text.lower().count(word)
            for word in uncertainty_words
        )
    )

    return df


# ---------------------------------------------------------
# Save Dataset
# ---------------------------------------------------------

def save_dataset():

    df = build_dataset()

    output_path = os.path.join(
        "data",
        "processed",
        "verification_dataset.csv"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Dataset created successfully: {output_path}"
    )

    print(
        f"Number of records: {len(df)}"
    )

    print("\nClass distribution:")

    print(
        df["needs_human_review"]
        .value_counts()
        .sort_index()
    )

    print("\nCategory distribution:")

    print(
        df["category"]
        .value_counts()
    )

    print("\nColumns:")

    print(
        list(df.columns)
    )


if __name__ == "__main__":
    save_dataset()