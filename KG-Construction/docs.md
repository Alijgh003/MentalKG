# Knowledge Graph (KG) Construction - Progress & Notes

## Completed
1. Assigned unique IDs to all nodes. ✅

## Next Steps
2. Build graph paths for each leaf node (to be used in KG construction).   Done.✅
2.1. Excluding none content parts... ✅
3. Test KG construction on the built tree nodes. ✅  
4. Run KG construction (two-phase entity & relation extraction) using **Huey**.   ✅
5. Store the constructed KG and create a backup.  
   - **Important:** Track time and resource usage of KG construction using **gemma4-31B** model. ✅

5.1 concat the last and first paragraphs two continuing pages and build a new paragraph and create KG ent and rel for each one...

## Storage
6. KG may be stored in a database like **Neo4j** or similar.


## Additional KG Construction Tasks
7. **Entity Mapping**  
   - 7.1. Use mini-batch clustering to map potentially duplicate entities/relations, keeping non-representative entries as aliases.  
   - 7.2. Repeat clustering with mini-batches until completion (may not be required depending on scale).


## Code Organization
8. Organize all of the KG construction code...


## Pending Questions
9. Should a smaller model like **gemma4-4BE** be used for KG construction?  
   - Decide whether to run on **server31** or via **OpenRouter**.





# Knowledge Graph (KG) Construction - Limitations

## Tree Construction
1. Since, We have not used OCR in order to extract the text of each page of the book,(we used simple PDF text extractor and a simple rule based MD construction wroten by deepseek)
      it is possible for our Tree to be not complete... for example: 
      ```
      Some symptoms not specifically mentioned in the DSM-5-TR criteria are also worth noting
         here.
         1. Even during an acute manic episode, many patients have brief periods of depression.
         These “microdepressions” are relatively common; depending on the symptoms associated
         with them, they may suggest that the specifier with mixed features is appropriate (p. 161).
         2. Patients may use substances (especially alcohol) in an attempt to relieve the
         uncomfortable, driven feeling that accompanies a severe manic episode. Less often, the
         substance use temporarily obscures the symptoms of the mood episode. When clinicians
         become confused about whether the substance use or the mania came first, the question
         can usually be sorted out with the help of informants.
         3. Catatonic symptoms occasionally occur during a manic episode, sometimes causing the
         episode to resemble schizophrenia. But a history (obtained from informants) of acute onset
         and previous episodes with recovery can help clarify the diagnosis. Then the specifier with
         catatonic features may be indicated (p. 100).
      ```
      We usually expect to see three bullet points as a child node for something like(Neurodevelopmental Disorder -> Symptoms not Included in the original DSM5-TR), but we've missed it.

2. There is a limiation on connecting different pages, especially when two following page are the continue of one another, we have tried to add a link between connected pages, but it would definitely not perfect.

3. Tables, may be a disaster...