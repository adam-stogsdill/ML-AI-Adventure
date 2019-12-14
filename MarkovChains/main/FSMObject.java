import java.util.Random;

/**
 * Finite State Machine Object
 *
 * The goal is to give this object finite state machine operations containing a certain amount of
 * Markov Chains given a certain initialization pattern and starting state.
 *
 * Using:
 * http://setosa.io/ev/markov-chains/
 * as a resource.
 */

public class FSMObject {

    private MarkovObject markovObject;
    private int initialStateIndex;
    private int currentStateIndex;

    public FSMObject(MarkovObject markovObject, String initialState){
        this.markovObject = markovObject;
        this.initialStateIndex = this.currentStateIndex = markovObject.getStateNameIndex(initialState);
    }

    public int getNextState() throws ProbabilityVectorException {
        ProbabilityVector pv = this.markovObject.getProbabilityVectorByIndex(this.currentStateIndex);
        for(int i = 0; i < pv.scalarMultiplier(10).length(); i++){

        }
    }

}
