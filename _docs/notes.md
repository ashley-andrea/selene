i am building an agentic system to help women understand the possible effects of different birth control pills on their body before taking them.
more detailes on the project can be found here #file:doc.md #file:tech.md. right now i need to take care of the main important stuff: the data.

so the idea is 3 main datasets:

- one for the pills, describing their type, what tehy are made of, stuff like that, and the contraindications -> this is real data;
- one for the patients: data on women modeling demographics, habits (kie smoker / non smoker), pathologies, everything that may incluence how a birth control pill reacts with their body -> this is synthetic!! we can use synthea (https://github.com/synthetichealth/synthea?utm_source=chatgpt.com) to generate data using only the modules that are of interest to us / creating a custom module in order to build a pool of candidates to train models on;
- finally the main dataset: symptom diaries -> this is the most important parta dn it will be synthetic as well, combining the user data with the possible medicines from the pill dataset, and describing how interactions work and what sympyoms are developed, how mood / conditions change, and things like that. this last one will be our benchmark for running simulations.

we need these 3 datasets to then create 2 ML models: one for categorizing users into profiles (from a list of characteristics indicated by the user, the model should inference which classes they like belong to and with what score, and associate blocking rules to each specific profile), the other for simulating long term effects of a pill on REAL user data (matches the pill's characteristics with the real user characteristics, not with the given category!!).
