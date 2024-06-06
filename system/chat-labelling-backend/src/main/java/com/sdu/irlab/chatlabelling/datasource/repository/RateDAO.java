package com.sdu.irlab.chatlabelling.datasource.repository;

import com.sdu.irlab.chatlabelling.common.ConversationStatus;
import com.sdu.irlab.chatlabelling.datasource.domain.Conversation;
import com.sdu.irlab.chatlabelling.datasource.domain.User;
import com.sdu.irlab.chatlabelling.datasource.domain.Rate;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface RateDAO extends JpaRepository<Rate, Long> {

}