/**
 * Markov Object will store a matrix and will store the information like the names of the state.
 */
public class MarkovObject {

    private String[] vectorNames;
    private ProbabilityVector[] probabilityVectors;

    public MarkovObject(String[] vectorNames, ProbabilityVector[] probabilityVectors){
        this.vectorNames = vectorNames;
        this.probabilityVectors = probabilityVectors;
    }

    public String[] getVectorNames(){
        return this.vectorNames;
    }

    public ProbabilityVector[] getProbabilityVectors(){
        return this.probabilityVectors;
    }

    public ProbabilityVector getProbabilityVectorByIndex(int index){
        return this.probabilityVectors[index];
    }

    public int getStateNameIndex(String param){
        for(int i = 0; i < this.vectorNames.length; i++){
            if(this.vectorNames[i].equals(param))
                return i;
        }
        // Return -1 if the value is not found in the vectorNames array.
        return -1;
    }

}
