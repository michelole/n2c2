package at.medunigraz.imi.bst.n2c2.classifier.factory;

import at.medunigraz.imi.bst.n2c2.classifier.Classifier;
import at.medunigraz.imi.bst.n2c2.model.Criterion;
import at.medunigraz.imi.bst.n2c2.nn.BILSTMEmbeddingsClassifier;

public class NNClassifierFactory implements ClassifierFactory {

    private static final Classifier classifier = new BILSTMEmbeddingsClassifier();

	@Override
	public Classifier getClassifier(Criterion criterion) {
		return classifier;
	}
}
