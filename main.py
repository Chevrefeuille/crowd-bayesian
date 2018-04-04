from model import estimator
import model.tools
from model.compute_pdfs import train
import numpy as np

if __name__ == "__main__":
    
    train_ratio = 0.3
    alphas = [0, 0.5, 1]
    for a in alphas:
        dry_rights, koi_rights = [], []
        for epoch in range(20):
            dry_set, koi_set = model.tools.get_data_set('data/classes/')
            dry_train, dry_test, koi_train, koi_test = model.tools.shuffle_data_set(dry_set, koi_set, train_ratio)
            train(dry_train, koi_train)

            pdf_dry_d, pdf_koi_d = np.load('data/pdfs/pdf_D_dry.dat'), np.load('data/pdfs/pdf_D_koi.dat')
            pdf_dry_grp_v, pdf_koi_grp_v = np.load('data/pdfs/pdf_vG_dry.dat'), np.load('data/pdfs/pdf_vG_koi.dat')
            pdf_dry_veldiff, pdf_koi_veldiff = np.load('data/pdfs/pdf_vDiff_dry.dat'), np.load('data/pdfs/pdf_vDiff_koi.dat')
            pdf_dry_heightdiff, pdf_koi_heightdiff = np.load('data/pdfs/pdf_HDiff_dry.dat'), np.load('data/pdfs/pdf_HDiff_koi.dat')

            dry_wrong, dry_right, koi_wrong, koi_right = estimator.compute_accuracy(dry_test, koi_test, pdf_dry_d, pdf_koi_d, pdf_dry_grp_v, pdf_koi_grp_v, pdf_dry_veldiff, pdf_koi_veldiff, pdf_dry_heightdiff, pdf_koi_heightdiff, a)
            dry_rights += [dry_right]
            koi_rights += [koi_right]

            print('-------------------------------')
            print('\t Right \t Wrong \t Rate\n')
            print('Dry\t {} \t {} \t {}'.format(dry_right, dry_wrong, dry_right/(dry_right + dry_wrong)))
            print('Koi\t {} \t {} \t {}'.format(koi_right, koi_wrong, koi_right/(koi_right + koi_wrong)))
            print('-------------------------------')

        m_suc_dry = np.mean(dry_rights) / len(dry_test)
        sdt_suc_dry = np.std(dry_rights) / len(dry_test)
        m_suc_koi = np.mean(koi_rights)/ len(koi_test)
        sdt_suc_koi = np.std(koi_rights) / len(koi_test)


        print('alpha = {}'.format(a))
        print('Doryo \t {} pm {}'.format(m_suc_dry, sdt_suc_dry))
        print('Koibito \t {} pm {}'.format(m_suc_koi, sdt_suc_koi))
        
        


