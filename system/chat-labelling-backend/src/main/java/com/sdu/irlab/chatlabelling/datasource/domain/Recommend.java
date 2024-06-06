package com.sdu.irlab.chatlabelling.datasource.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;

import javax.persistence.Entity;
import javax.persistence.EnumType;
import javax.persistence.Enumerated;

import javax.persistence.*;
import java.util.List;

@Entity
public class Recommend extends BaseEntity {
    /**
    * 保存推荐信息，防止网络出现故障，无法恢复
    **/

    @JsonIgnore
    @ManyToOne(cascade = {CascadeType.MERGE, CascadeType.REFRESH}, optional = false)
    private User user; // 评价用户的ID

    @JsonIgnore
    @ManyToOne(cascade = {CascadeType.MERGE, CascadeType.REFRESH}, optional = false)
    private Conversation conversation; // 对话的ID

    private String evaluationObject; // 评价对象

    private String nameID;  // 商品qid

    public String getNameID() {
        return nameID;
    }

    public void setNameID(String nameID) {
        this.nameID = nameID;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public Conversation getConversation() {
        return conversation;
    }

    public void setConversation(Conversation conversation) {
        this.conversation = conversation;
    }

    public String getEvaluationObject() {
        return evaluationObject;
    }

    public void setEvaluationObject(String evaluationObject) {
        this.evaluationObject = evaluationObject;
    }

}
